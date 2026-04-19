//+------------------------------------------------------------------+
//| TitanEA_V3.mq5 — Titan V3.0 Expert Advisor                      |
//| v3.3: 500-tick Trend Guard — gates signals against macro drift   |
//|       Alpha Pivot v3.4: 3-pip TP, 100-tick exit, real spread     |
//+------------------------------------------------------------------+
#property copyright "Titan HFT Systems"
#property version   "3.30"
#property strict

#include <Trade\Trade.mqh>
#include <Math\Stat\Math.mqh>

#resource "\\Files\\TitanV3.onnx" as uchar g_onnx_buf[]

input group "=== MODEL ==="
input double  InpLong    = 0.4029; // Long conviction threshold (top 3% — calibrated Apr 19)
input double  InpShort   = 0.0441; // Short conviction threshold (bottom 3% — calibrated Apr 19)
input int     InpMagic   = 30000;

input group "=== TREND GUARD ==="
input int     InpDriftWin  = 500;  // Macro drift window (ticks)
input double  InpDriftPips = 1.5;  // Drift threshold in pips — above spread cost

input group "=== RISK ==="
input double  InpKelly   = 0.02;   // Fraction of equity per trade
input double  InpDailyDD = 0.02;   // Daily drawdown kill switch (2%)
input double  InpSpreadS = 2.0;    // Spread guard (sigma multiples)
input int     InpMaxPos  = 1;      // Max simultaneous positions

input group "=== EXIT ==="
input double  InpTP_Pips = 3.0;    // Take profit in pips — Alpha Pivot v3.4 (3-pip barrier)
input double  InpSL_Pips = 0.0;    // Stop loss in pips (0 = time-exit only, matches MTB framework)
input int     InpMaxTicks= 100;    // Time-exit: close after N ticks — Alpha Pivot v3.4

input group "=== FEATURES ==="
input int     InpN       = 128;
input int     InpShortW  = 16;
input int     InpSGCW    = 32;
input double  InpEps     = 1e-10;
input double  InpMinDt   = 1.0;
input double  InpClip    = 3.0;
input int     InpFlagBid = 2;
input int     InpFlagAsk = 4;

#define N_FEAT  16
#define SEQ_LEN 128

//── Rolling Z-Score ring-buffer (ddof=0, matches Python _zscore_clip) ──────
class CRollZ {
   double buf[]; int head,win; double sum,ssq,eps,clip; bool full;
public:
   void Init(int w,double e=1e-10,double c=3.0){
      win=w;eps=e;clip=c;head=0;sum=0;ssq=0;full=false;
      ArrayResize(buf,w);ArrayInitialize(buf,0);}
   double Push(double v){
      double old=buf[head];buf[head]=v;head=(head+1)%win;
      if(head==0)full=true;
      sum+=v-old;ssq+=v*v-old*old;
      if(!full)return 0.0;
      double mu=sum/win;
      double sig=MathSqrt(MathMax(ssq/win-mu*mu,0))+eps;
      double z=(v-mu)/sig;
      return MathMax(MathMin(z,clip),-clip)/clip;}
   bool IsFull(){return full;}
};

class CRing {
   double buf[]; int head,cnt,win;
public:
   void Init(int w){win=w;head=0;cnt=0;ArrayResize(buf,w);ArrayInitialize(buf,0);}
   void Push(double v){buf[head]=v;head=(head+1)%win;if(cnt<win)cnt++;}
   double At(int off)const{int i=(head-1-off+win*2)%win;return buf[i];}
   double Sum()const{double s=0;for(int i=0;i<cnt;i++)s+=buf[i];return s;}
   double Mean()const{return cnt>0?Sum()/cnt:0;}
   double StdDev()const{
      if(cnt<2)return 0;double m=Mean(),ss=0;
      for(int i=0;i<cnt;i++)ss+=(buf[i]-m)*(buf[i]-m);
      return MathSqrt(ss/cnt);}
   int Count()const{return cnt;}
   bool Full()const{return cnt>=win;}
};

CRing  g_mid,g_spread,g_dt,g_buy,g_sell,g_flags;
CRing  g_dt16,g_dt64,g_svel,g_sp_stat;
CRing  g_ptp_raw;
CRing  g_drift_mid;   // 500-tick macro drift buffer for Trend Guard
CRollZ g_zMVDI,g_zTWKJ,g_zSGC,g_zMFE;
CRollZ g_zTWAP,g_zMOM,g_zICE,g_zTCE,g_zH0,g_zH1;

double g_seq[SEQ_LEN][N_FEAT];
int    g_seqHead=0,g_seqCnt=0;

long   g_onnx=INVALID_HANDLE;
float  g_in[];
float  g_out[];

CTrade g_trade;
double g_navStart=0;
datetime g_lastDay=0;
bool   g_isTesting=false;

// Position tracking for time-based exit
ulong  g_posTicket=0;
int    g_ticksSinceEntry=0;

double g_prevMid=0,g_prevVel=0,g_prevAcc=0;
double g_prevSpread=0,g_prevSvel=0,g_prevMid2=0;
double g_prevMFE=0,g_prevDMFE=0;
double g_prevTms=0;

int OnInit(){
   g_isTesting=(bool)MQLInfoInteger(MQL_TESTER);

   g_mid.Init(InpN);g_spread.Init(InpN);g_dt.Init(InpN);
   g_buy.Init(InpN);g_sell.Init(InpN);g_flags.Init(InpN);
   g_dt16.Init(InpShortW);g_dt64.Init(64);
   g_svel.Init(InpSGCW);g_sp_stat.Init(200);
   g_ptp_raw.Init(4);
   g_drift_mid.Init(InpDriftWin);
   g_zMVDI.Init(InpN,InpEps,InpClip);g_zTWKJ.Init(InpN,InpEps,InpClip);
   g_zSGC.Init(InpN,InpEps,InpClip); g_zMFE.Init(InpN,InpEps,InpClip);
   g_zTWAP.Init(InpN,InpEps,InpClip);g_zMOM.Init(InpN,InpEps,InpClip);
   g_zICE.Init(InpN,InpEps,InpClip); g_zTCE.Init(InpN,InpEps,InpClip);
   g_zH0.Init(InpN,InpEps,InpClip);  g_zH1.Init(InpN,InpEps,InpClip);
   ArrayInitialize(g_seq,0.0);

   // ONNX
   g_onnx=OnnxCreateFromBuffer(g_onnx_buf,ONNX_DEFAULT);
   if(g_onnx==INVALID_HANDLE){
      Print("[TITAN] ONNX load FAILED: ",GetLastError());return INIT_FAILED;}
   ulong si[]={1,SEQ_LEN,N_FEAT},so[]={1,1};
   if(!OnnxSetInputShape(g_onnx,0,si)||!OnnxSetOutputShape(g_onnx,0,so)){
      Print("[TITAN] ONNX shape FAILED.");return INIT_FAILED;}
   ArrayResize(g_in,1*SEQ_LEN*N_FEAT);ArrayResize(g_out,1);

   // Trade settings — auto-adapt for Strategy Tester
   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(10);
   g_trade.SetTypeFilling(g_isTesting ? ORDER_FILLING_RETURN : ORDER_FILLING_IOC);
   g_trade.SetAsyncMode(false);   // must be false in tester

   g_navStart=AccountInfoDouble(ACCOUNT_EQUITY);
   g_lastDay=TimeCurrent();

   // Subscribe to DOM only in live (not available in tester)
   if(!g_isTesting)MarketBookAdd(_Symbol);

   Print("[TITAN V3.2] Init OK. Mode=",g_isTesting?"BACKTEST":"LIVE",
         " Features=",N_FEAT," NAV=",g_navStart);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason){
   if(!g_isTesting)MarketBookRelease(_Symbol);
   if(g_onnx!=INVALID_HANDLE)OnnxRelease(g_onnx);
   Print("[TITAN] Deinit reason=",reason);
}

//── Core tick processing — called from both OnTick and OnBookEvent ──────────
void ProcessTick(){
   datetime now=TimeCurrent();

   // Daily reset
   if(TimeDay(now)!=TimeDay(g_lastDay)){
      g_navStart=AccountInfoDouble(ACCOUNT_EQUITY);g_lastDay=now;}

   // Daily DD kill
   if(g_navStart>0 &&
      (AccountInfoDouble(ACCOUNT_EQUITY)-g_navStart)/g_navStart < -InpDailyDD){
      Print("[TITAN] Daily DD kill.");return;}

   // Time-based position exit (backup if TP/SL not hit)
   if(g_posTicket>0){
      g_ticksSinceEntry++;
      if(g_ticksSinceEntry>=InpMaxTicks){
         if(PositionSelectByTicket(g_posTicket)){
            g_trade.PositionClose(g_posTicket);
            Print("[TITAN] Time-exit at tick ",g_ticksSinceEntry);}
         g_posTicket=0;g_ticksSinceEntry=0;}}

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol,tick))return;
   double bid=tick.bid,ask=tick.ask,mid=(bid+ask)/2.0,spread=ask-bid;
   int    flags=(int)tick.flags;
   double tms=(double)tick.time_msc;

   double dt=(g_prevTms>0)?MathMax(tms-g_prevTms,InpMinDt):InpMinDt;
   g_prevTms=tms;

   g_mid.Push(mid);g_spread.Push(spread);g_dt.Push(dt);
   g_dt16.Push(dt);g_dt64.Push(dt);g_sp_stat.Push(spread);
   g_drift_mid.Push(mid);
   double ib=(flags==InpFlagAsk)?1.0:0.0;
   double is_=(flags==InpFlagBid)?1.0:0.0;
   g_buy.Push(ib);g_sell.Push(is_);g_flags.Push((double)flags);

   // Spread guard
   double spMu=g_sp_stat.Mean(),spSig=g_sp_stat.StdDev();
   if(spread>spMu+InpSpreadS*spSig+InpEps)return;
   if(!g_mid.Full())return;

   double f[N_FEAT];
   double BN=g_buy.Sum(),SN=g_sell.Sum();

   // F1 FDPI
   f[0]=(BN-SN)/(BN+SN+InpEps);

   // F2 MVDI
   double sSig=g_spread.StdDev(),sMu=MathAbs(g_spread.Mean())+InpEps;
   double spreadCV=sSig/sMu;
   f[1]=g_zMVDI.Push(spreadCV/(spreadCV*0.5+InpEps));

   // F3 TWKJ
   double vel=(mid-g_prevMid)/dt;
   double acc=(vel-g_prevVel)/dt;
   double jerk=(acc-g_prevAcc)/dt;
   f[2]=g_zTWKJ.Push(jerk);
   g_prevMid=mid;g_prevVel=vel;g_prevAcc=acc;

   // F4 QAD
   double dtSum=g_dt.Sum();
   double aRate=BN/(dtSum/100.0+InpEps),bRate=SN/(dtSum/100.0+InpEps);
   f[3]=MathMax(MathMin((aRate-bRate)/(aRate+bRate+InpEps),1),-1);

   // F5 SGC
   double sv=(spread-g_prevSpread)/dt,sa=(sv-g_prevSvel)/dt;
   g_svel.Push(sa);
   double sgcMin=g_svel.At(0);
   for(int k=1;k<InpSGCW;k++)if(g_svel.At(k)<sgcMin)sgcMin=g_svel.At(k);
   f[4]=g_zSGC.Push(sgcMin);
   g_prevSpread=spread;g_prevSvel=sv;

   // F6 HURST (variance-ratio approx)
   double stdN=g_mid.StdDev();
   f[5]=((MathMin(MathMax(0.5+(stdN-g_prevMid2)*10,0.0),1.0))-0.5)*2.0;
   g_prevMid2=stdN;

   // F7 TOPO_H0 (dt entropy approx)
   double dtCV=g_dt.StdDev()/(g_dt.Mean()+InpEps);
   f[6]=g_zH0.Push(-dtCV);

   // F8 TOPO_H1 (cycle proxy)
   f[7]=g_zH1.Push(MathAbs(f[0])*(1.0-dtCV));

   // F9 MFE
   double U=MathAbs(f[0]),T=MathMin(1000.0/dt,1000.0),S=dtCV;
   f[8]=g_zMFE.Push(U-T*S/1000.0);

   // F10 PTP
   double dF=f[8]-g_prevMFE,d2F=dF-g_prevDMFE;
   g_ptp_raw.Push((dF<0&&d2F<0)?1.0:0.0);
   f[9]=(g_ptp_raw.Mean()-0.5)*2.0;
   g_prevMFE=f[8];g_prevDMFE=dF;

   // F11 TWAP_PROB
   f[10]=g_zTWAP.Push(1.0/(1.0+dtCV));

   // F12 MOM_IGNITE
   double buy16=BN*(double)InpShortW/InpN;
   double sel16=SN*(double)InpShortW/InpN;
   f[11]=g_zMOM.Push(buy16/(BN+InpEps)-sel16/(SN+InpEps));

   // F13 ICE_SCORE
   bool mstatic=MathAbs(mid-g_prevMid)<_Point;
   bool fconsist=((int)g_flags.At(0)==(int)g_flags.At(1));
   f[12]=g_zICE.Push((mstatic&&fconsist)?1.0:0.0);

   // F14 TCE
   f[13]=g_zTCE.Push(-dtCV);

   // F15/F16 Session encoding
   MqlDateTime tm2;TimeToStruct(now,tm2);
   double angle=2.0*M_PI*tm2.hour/24.0;
   f[14]=MathSin(angle);
   f[15]=MathCos(angle);

   // Update sequence ring buffer
   for(int fi=0;fi<N_FEAT;fi++)g_seq[g_seqHead][fi]=f[fi];
   g_seqHead=(g_seqHead+1)%SEQ_LEN;
   if(g_seqCnt<SEQ_LEN)g_seqCnt++;
   if(g_seqCnt<SEQ_LEN)return;

   // Flatten in chronological order for ONNX
   for(int t=0;t<SEQ_LEN;t++){
      int row=(g_seqHead+t)%SEQ_LEN;
      for(int fi=0;fi<N_FEAT;fi++)
         g_in[t*N_FEAT+fi]=(float)g_seq[row][fi];}

   if(!OnnxRun(g_onnx,ONNX_NO_CONVERSION,g_in,g_out)){
      Print("[TITAN] Inference error: ",GetLastError());return;}
   double conv=(double)g_out[0];

   // Skip if already in a position
   if(PositionsTotal()>=InpMaxPos)return;

   // ── Trend Guard: gate signals against 500-tick macro drift ──────────────
   bool allow_long=true, allow_short=true;
   if(g_drift_mid.Full()){
      double pipSize0=_Point*10;
      double oldest=g_drift_mid.At(InpDriftWin-1);
      double drift_pips=(mid-oldest)/pipSize0;
      if(drift_pips >  InpDriftPips) allow_short=false;  // uptrend: no shorts
      if(drift_pips < -InpDriftPips) allow_long =false;  // downtrend: no longs
      if(MathAbs(drift_pips)>InpDriftPips)
         Print("[TITAN] TrendGuard drift=",DoubleToString(drift_pips,2),
               "pip allow_long=",allow_long," allow_short=",allow_short);}

   // Lot sizing: fixed minimum lot when SL=0 (time-exit only, MTB framework)
   double equity=AccountInfoDouble(ACCOUNT_EQUITY);
   double tv=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE);
   double ts=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
   double minL=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   double step=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   double pipSize=_Point*10;
   double pipVal=(ts>0)?(tv/ts)*pipSize:tv*pipSize;
   double lot;
   if(InpSL_Pips>InpEps){
      double rawLot=(pipVal>InpEps)?(equity*InpKelly)/(InpSL_Pips*pipVal):minL;
      lot=MathMax(MathRound(rawLot/step)*step,minL);}
   else{
      lot=minL;}  // SL=0: use minimum lot, rely on time-exit

   // SL/TP in price (SL=0 means no hard stop, broker needs a value so use 0)
   double tpDist=InpTP_Pips*pipSize;
   double slDistP=InpSL_Pips*pipSize;

   if(conv>=InpLong && allow_long){
      double sl=(slDistP>InpEps)?NormalizeDouble(ask-slDistP,_Digits):0;
      double tp=NormalizeDouble(ask+tpDist,_Digits);
      Print("[TITAN] LONG conv=",DoubleToString(conv,4),
            " lot=",lot," TP=",tp," drift_gated=OK");
      if(g_trade.Buy(lot,_Symbol,ask,sl,tp,"TitanV3_Long")){
         g_posTicket=g_trade.ResultDeal();
         g_ticksSinceEntry=0;}}
   else if(conv<=InpShort && allow_short){
      double sl=(slDistP>InpEps)?NormalizeDouble(bid+slDistP,_Digits):0;
      double tp=NormalizeDouble(bid-tpDist,_Digits);
      Print("[TITAN] SHORT conv=",DoubleToString(conv,4),
            " lot=",lot," TP=",tp," drift_gated=OK");
      if(g_trade.Sell(lot,_Symbol,bid,sl,tp,"TitanV3_Short")){
         g_posTicket=g_trade.ResultDeal();
         g_ticksSinceEntry=0;}}
}

//── Event handlers ──────────────────────────────────────────────────────────

// OnTick fires in BOTH live and Strategy Tester — primary handler
void OnTick(){ ProcessTick(); }

// OnBookEvent fires in live only (DOM changes) — extra precision in live
void OnBookEvent(const string &sym){
   if(sym==_Symbol && !g_isTesting)ProcessTick();}

int TimeDay(datetime t){MqlDateTime s;TimeToStruct(t,s);return s.day;}
