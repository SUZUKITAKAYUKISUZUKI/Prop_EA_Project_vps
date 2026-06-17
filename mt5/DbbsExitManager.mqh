//+------------------------------------------------------------------+
//| DbbsExitManager.mqh — DBBS H1 BB20 ±1σ trailing exit (Live)      |
//+------------------------------------------------------------------+
#ifndef DBBS_EXIT_MANAGER_MQH
#define DBBS_EXIT_MANAGER_MQH

#define DBBS_EXIT_MAX_TRACKS 16
#define DBBS_BB_CACHE_MAX   8

struct DbbsExitTrack
{
   bool     active;
   ulong    ticket;
   string   symbol;
   long     direction;
   double   entry;
   double   initial_sl;
   int      min_hold_h1;
   int      max_hold_h1;
   int      bars_held_h1;
   datetime last_h1_bar;
};

DbbsExitTrack g_dbbs_tracks[DBBS_EXIT_MAX_TRACKS];

struct DbbsBbCacheEntry
{
   string symbol;
   int    handle;
};

DbbsBbCacheEntry g_dbbs_bb_cache[DBBS_BB_CACHE_MAX];
datetime         g_dbbs_last_h1_bar_time[DBBS_BB_CACHE_MAX];

//+------------------------------------------------------------------+
int DbbsExit_FindBbCacheIndex(const string symbol)
{
   for(int i = 0; i < DBBS_BB_CACHE_MAX; i++)
   {
      if(g_dbbs_bb_cache[i].symbol == symbol)
         return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
int DbbsExit_AcquireBbCacheIndex(const string symbol)
{
   int idx = DbbsExit_FindBbCacheIndex(symbol);
   if(idx >= 0)
      return idx;

   for(int i = 0; i < DBBS_BB_CACHE_MAX; i++)
   {
      if(g_dbbs_bb_cache[i].symbol == "")
      {
         g_dbbs_bb_cache[i].symbol = symbol;
         g_dbbs_bb_cache[i].handle = INVALID_HANDLE;
         g_dbbs_last_h1_bar_time[i] = 0;
         return i;
      }
   }
   return -1;
}

//+------------------------------------------------------------------+
int DbbsExit_GetBandsHandle(const string symbol)
{
   int idx = DbbsExit_AcquireBbCacheIndex(symbol);
   if(idx < 0)
      return INVALID_HANDLE;

   if(g_dbbs_bb_cache[idx].handle == INVALID_HANDLE)
   {
      g_dbbs_bb_cache[idx].handle = iBands(symbol, PERIOD_H1, 20, 0, 2.0, PRICE_CLOSE);
      if(g_dbbs_bb_cache[idx].handle == INVALID_HANDLE)
         Print("DbbsExit: iBands create failed symbol=", symbol, " err=", GetLastError());
   }
   return g_dbbs_bb_cache[idx].handle;
}

//+------------------------------------------------------------------+
void DbbsExit_Deinit()
{
   for(int i = 0; i < DBBS_BB_CACHE_MAX; i++)
   {
      if(g_dbbs_bb_cache[i].handle != INVALID_HANDLE)
      {
         IndicatorRelease(g_dbbs_bb_cache[i].handle);
         g_dbbs_bb_cache[i].handle = INVALID_HANDLE;
      }
      g_dbbs_bb_cache[i].symbol = "";
      g_dbbs_last_h1_bar_time[i] = 0;
   }
}

//+------------------------------------------------------------------+
bool DbbsExit_SymbolHasActiveTrack(const string symbol)
{
   for(int i = 0; i < DBBS_EXIT_MAX_TRACKS; i++)
   {
      if(g_dbbs_tracks[i].active && g_dbbs_tracks[i].symbol == symbol)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
int DbbsExit_FindSlot()
{
   for(int i = 0; i < DBBS_EXIT_MAX_TRACKS; i++)
   {
      if(!g_dbbs_tracks[i].active)
         return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
int DbbsExit_FindByTicket(const ulong ticket)
{
   for(int i = 0; i < DBBS_EXIT_MAX_TRACKS; i++)
   {
      if(g_dbbs_tracks[i].active && g_dbbs_tracks[i].ticket == ticket)
         return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
void DbbsExit_ClearSlot(const int idx)
{
   g_dbbs_tracks[idx].active = false;
   g_dbbs_tracks[idx].ticket = 0;
}

//+------------------------------------------------------------------+
bool DbbsExit_ClosePosition(const ulong ticket, const string symbol, const string reason)
{
   if(!PositionSelectByTicket(ticket))
      return false;

   long pos_type = PositionGetInteger(POSITION_TYPE);
   double volume = PositionGetDouble(POSITION_VOLUME);
   if(volume <= 0.0)
      return false;

   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action    = TRADE_ACTION_DEAL;
   request.position  = ticket;
   request.symbol    = symbol;
   request.volume    = volume;
   request.deviation = 30;
   request.magic     = PositionGetInteger(POSITION_MAGIC);
   request.comment   = "DbbsExit";

   if(pos_type == POSITION_TYPE_BUY)
   {
      request.type  = ORDER_TYPE_SELL;
      request.price = SymbolInfoDouble(symbol, SYMBOL_BID);
   }
   else
   {
      request.type  = ORDER_TYPE_BUY;
      request.price = SymbolInfoDouble(symbol, SYMBOL_ASK);
   }

   if(!OrderSend(request, result))
   {
      Print("DbbsExit close failed ticket=", ticket, " retcode=", result.retcode, " ", result.comment);
      return false;
   }

   Print("DbbsExit closed ticket=", ticket, " reason=", reason);
   return true;
}

//+------------------------------------------------------------------+
bool DbbsExit_Bb20Upper1Sigma(
   const string symbol,
   const int bar_shift,
   double &upper1,
   double &lower1,
   double &close_h1
)
{
   int handle = DbbsExit_GetBandsHandle(symbol);
   if(handle == INVALID_HANDLE)
      return false;

   double upper[], middle[], lower[];
   ArraySetAsSeries(upper, true);
   ArraySetAsSeries(middle, true);
   ArraySetAsSeries(lower, true);

   if(CopyBuffer(handle, 0, bar_shift, 1, upper) != 1)
      return false;
   if(CopyBuffer(handle, 1, bar_shift, 1, middle) != 1)
      return false;
   if(CopyBuffer(handle, 2, bar_shift, 1, lower) != 1)
      return false;

   double sigma = (upper[0] - middle[0]) / 2.0;
   upper1 = middle[0] + sigma;
   lower1 = middle[0] - sigma;

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(symbol, PERIOD_H1, bar_shift, 1, rates) != 1)
      return false;
   close_h1 = rates[0].close;
   return true;
}

//+------------------------------------------------------------------+
void DbbsExit_Register(
   const ulong ticket,
   const string symbol,
   const long direction,
   const double entry,
   const double initial_sl,
   const int min_hold_h1,
   const int max_hold_h1
)
{
   int idx = DbbsExit_FindByTicket(ticket);
   if(idx < 0)
   {
      idx = DbbsExit_FindSlot();
      if(idx < 0)
      {
         Print("DbbsExit: track table full, ticket=", ticket);
         return;
      }
   }

   g_dbbs_tracks[idx].active        = true;
   g_dbbs_tracks[idx].ticket        = ticket;
   g_dbbs_tracks[idx].symbol        = symbol;
   g_dbbs_tracks[idx].direction     = direction;
   g_dbbs_tracks[idx].entry         = entry;
   g_dbbs_tracks[idx].initial_sl    = initial_sl;
   g_dbbs_tracks[idx].min_hold_h1   = min_hold_h1;
   g_dbbs_tracks[idx].max_hold_h1   = max_hold_h1;
   g_dbbs_tracks[idx].bars_held_h1  = 0;
   g_dbbs_tracks[idx].last_h1_bar   = 0;

   Print(
      "DbbsExit registered ticket=", ticket,
      " min_hold_h1=", min_hold_h1,
      " max_hold_h1=", max_hold_h1
   );
}

//+------------------------------------------------------------------+
double DbbsExit_FloatingR(const DbbsExitTrack &t, const double close_h1)
{
   double risk = MathAbs(t.entry - t.initial_sl);
   if(risk <= 0.0)
      return 0.0;
   double pnl = (t.direction == POSITION_TYPE_BUY)
      ? (close_h1 - t.entry)
      : (t.entry - close_h1);
   return pnl / risk;
}

//+------------------------------------------------------------------+
void DbbsExit_ProcessH1Bar(const int idx, const double close_h1, const double upper1, const double lower1)
{
   DbbsExitTrack t = g_dbbs_tracks[idx];
   if(!PositionSelectByTicket(t.ticket))
   {
      DbbsExit_ClearSlot(idx);
      return;
   }

   g_dbbs_tracks[idx].bars_held_h1++;

   if(t.direction == POSITION_TYPE_BUY && close_h1 <= t.initial_sl)
   {
      DbbsExit_ClosePosition(t.ticket, t.symbol, "SL");
      DbbsExit_ClearSlot(idx);
      return;
   }
   if(t.direction == POSITION_TYPE_SELL && close_h1 >= t.initial_sl)
   {
      DbbsExit_ClosePosition(t.ticket, t.symbol, "SL");
      DbbsExit_ClearSlot(idx);
      return;
   }

   if(g_dbbs_tracks[idx].bars_held_h1 >= t.max_hold_h1)
   {
      DbbsExit_ClosePosition(t.ticket, t.symbol, "MAX_HOLD");
      DbbsExit_ClearSlot(idx);
      return;
   }

   if(g_dbbs_tracks[idx].bars_held_h1 < t.min_hold_h1)
      return;

   double float_r = DbbsExit_FloatingR(t, close_h1);
   if(float_r <= -1.0)
   {
      DbbsExit_ClosePosition(t.ticket, t.symbol, "SL_R_FLOOR");
      DbbsExit_ClearSlot(idx);
      return;
   }

   if(t.direction == POSITION_TYPE_BUY && close_h1 < upper1)
   {
      DbbsExit_ClosePosition(t.ticket, t.symbol, "TRAIL_BB20");
      DbbsExit_ClearSlot(idx);
      return;
   }

   if(t.direction == POSITION_TYPE_SELL && close_h1 > lower1)
   {
      DbbsExit_ClosePosition(t.ticket, t.symbol, "TRAIL_BB20");
      DbbsExit_ClearSlot(idx);
   }
}

//+------------------------------------------------------------------+
void DbbsExit_TryRegisterFromSignal(
   const ulong ticket,
   const string symbol,
   const long direction,
   const double entry,
   const double initial_sl,
   const string response_json
)
{
   string setup_type, exit_mode;
   if(!ExtractJsonString(response_json, "setup_type", setup_type))
      return;
   if(setup_type != "DBBS")
      return;
   if(!ExtractJsonString(response_json, "exit_mode", exit_mode))
      return;
   if(exit_mode != "DBBS_TRAIL")
      return;

   int min_hold = 3;
   int max_hold = 48;
   DbbsExit_ExtractInt(response_json, "exit_min_hold_h1", min_hold);
   DbbsExit_ExtractInt(response_json, "exit_max_hold_h1", max_hold);

   DbbsExit_Register(ticket, symbol, direction, entry, initial_sl, min_hold, max_hold);
}

//+------------------------------------------------------------------+
bool DbbsExit_ExtractInt(const string json, const string key, int &value)
{
   string pattern = "\"" + key + "\":";
   int pos = StringFind(json, pattern);
   if(pos < 0)
      return false;
   pos += StringLen(pattern);
   int end = pos;
   while(end < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, end);
      if(ch == ',' || ch == '}' || ch == ']')
         break;
      end++;
   }
   value = (int)StringToInteger(StringSubstr(json, pos, end - pos));
   return true;
}

//+------------------------------------------------------------------+
void DbbsExit_PurgeClosed()
{
   for(int i = 0; i < DBBS_EXIT_MAX_TRACKS; i++)
   {
      if(!g_dbbs_tracks[i].active)
         continue;
      if(!PositionSelectByTicket(g_dbbs_tracks[i].ticket))
         DbbsExit_ClearSlot(i);
   }
}

//+------------------------------------------------------------------+
void DbbsExit_OnNewH1Bar(const string symbol)
{
   if(!DbbsExit_SymbolHasActiveTrack(symbol))
      return;

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(symbol, PERIOD_H1, 1, 1, rates) != 1)
      return;

   datetime bar_time = rates[0].time;
   int cache_idx = DbbsExit_FindBbCacheIndex(symbol);
   if(cache_idx >= 0 && g_dbbs_last_h1_bar_time[cache_idx] == bar_time)
      return;

   double upper1 = 0.0, lower1 = 0.0, close_h1 = rates[0].close;
   if(!DbbsExit_Bb20Upper1Sigma(symbol, 1, upper1, lower1, close_h1))
      return;

   if(cache_idx >= 0)
      g_dbbs_last_h1_bar_time[cache_idx] = bar_time;

   for(int i = 0; i < DBBS_EXIT_MAX_TRACKS; i++)
   {
      if(!g_dbbs_tracks[i].active)
         continue;
      if(g_dbbs_tracks[i].symbol != symbol)
         continue;
      if(g_dbbs_tracks[i].last_h1_bar == bar_time)
         continue;

      g_dbbs_tracks[i].last_h1_bar = bar_time;
      DbbsExit_ProcessH1Bar(i, close_h1, upper1, lower1);
   }
}

//+------------------------------------------------------------------+
void DbbsExit_ManageOpenPositions(const ulong magic)
{
   DbbsExit_PurgeClosed();

   string symbols[DBBS_BB_CACHE_MAX];
   int symbol_count = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)magic)
         continue;

      string symbol = PositionGetString(POSITION_SYMBOL);
      bool found = false;
      for(int j = 0; j < symbol_count; j++)
      {
         if(symbols[j] == symbol)
         {
            found = true;
            break;
         }
      }
      if(!found && symbol_count < DBBS_BB_CACHE_MAX)
      {
         symbols[symbol_count] = symbol;
         symbol_count++;
      }
   }

   for(int k = 0; k < symbol_count; k++)
      DbbsExit_OnNewH1Bar(symbols[k]);
}

#endif // DBBS_EXIT_MANAGER_MQH
