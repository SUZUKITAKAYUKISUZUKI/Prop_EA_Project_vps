//+------------------------------------------------------------------+
//| PropEA_WebRequestLock.mqh — MT5 allows one WebRequest at a time   |
//+------------------------------------------------------------------+
#ifndef PROPEA_WEBREQUEST_LOCK_MQH
#define PROPEA_WEBREQUEST_LOCK_MQH

#define PROPEA_WR_LOCK_GV "PropEA_WebRequest_Lock"
#define PROPEA_WR_LOCK_TTL_SEC 120

static bool g_propea_wr_local_busy = false;

//+------------------------------------------------------------------+
bool PropEA_TryAcquireWebRequestLock()
{
   if(g_propea_wr_local_busy)
      return false;

   if(GlobalVariableCheck(PROPEA_WR_LOCK_GV))
   {
      datetime locked_at = (datetime)GlobalVariableGet(PROPEA_WR_LOCK_GV);
      if(TimeCurrent() - locked_at < PROPEA_WR_LOCK_TTL_SEC)
         return false;
      GlobalVariableDel(PROPEA_WR_LOCK_GV);
   }

   g_propea_wr_local_busy = true;
   GlobalVariableSet(PROPEA_WR_LOCK_GV, (double)TimeCurrent());
   return true;
}

//+------------------------------------------------------------------+
void PropEA_ReleaseWebRequestLock()
{
   g_propea_wr_local_busy = false;
   if(GlobalVariableCheck(PROPEA_WR_LOCK_GV))
      GlobalVariableDel(PROPEA_WR_LOCK_GV);
}

#endif
