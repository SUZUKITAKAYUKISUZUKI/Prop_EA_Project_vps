//+------------------------------------------------------------------+
//| CspaExitManager.mqh — CSPA 建値ストップ + ATR トレーリング (Live)   |
//| Python strategies/cspa.py::track_cspa_trade_outcome と同一ルール   |
//+------------------------------------------------------------------+
#ifndef CSPA_EXIT_MANAGER_MQH
#define CSPA_EXIT_MANAGER_MQH

#define CSPA_EXIT_MAX_TRACKS 16
#define FINTOKEI_COMMISSION_RT_USD 6.0
#define FINTOKEI_MIN_NET_PROFIT_USD 0.50

struct CspaExitTrack
{
   ulong    ticket;
   string   symbol;
   long     direction;       // POSITION_TYPE_BUY / SELL
   double   entry;
   double   initial_sl;
   double   take_profit;
   double   atr;
   double   lot_size;
   bool     be_enabled;
   bool     trail_enabled;
   double   be_arm_mfe_r;
   double   be_trigger_mfe_r;
   double   be_pullback_close_r;
   int      be_rhythm_max_bars;
   double   trail_atr_mult;
   double   be_buffer_atr;
   double   current_sl;
   double   peak_favorable;
   bool     extension_armed;
   bool     sl_at_breakeven;
   int      bars_since_entry;
   datetime last_bar_time;
   bool     active;
};

CspaExitTrack g_cspa_tracks[CSPA_EXIT_MAX_TRACKS];

//+------------------------------------------------------------------+
int CspaExit_FindSlot()
{
   for(int i = 0; i < CSPA_EXIT_MAX_TRACKS; i++)
   {
      if(!g_cspa_tracks[i].active)
         return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
int CspaExit_FindByTicket(const ulong ticket)
{
   for(int i = 0; i < CSPA_EXIT_MAX_TRACKS; i++)
   {
      if(g_cspa_tracks[i].active && g_cspa_tracks[i].ticket == ticket)
         return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
void CspaExit_ClearSlot(const int idx)
{
   g_cspa_tracks[idx].active = false;
   g_cspa_tracks[idx].ticket = 0;
}

//+------------------------------------------------------------------+
double CspaExit_InitialRisk(const CspaExitTrack &t)
{
   return MathAbs(t.entry - t.initial_sl);
}

//+------------------------------------------------------------------+
double CspaExit_ProfitR(const CspaExitTrack &t, const double price)
{
   double risk = CspaExit_InitialRisk(t);
   if(risk <= 0.0)
      return 0.0;
   if(t.direction == POSITION_TYPE_BUY)
      return (price - t.entry) / risk;
   return (t.entry - price) / risk;
}

//+------------------------------------------------------------------+
double CspaExit_RatchetSl(const CspaExitTrack &t, const double current_sl, const double new_sl)
{
   if(t.direction == POSITION_TYPE_BUY)
      return MathMax(current_sl, new_sl);
   return MathMin(current_sl, new_sl);
}

//+------------------------------------------------------------------+
double CspaExit_CommissionSlBuffer(const string symbol, const double lot)
{
   if(lot <= 0.0)
      return 0.0;
   double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tick_size <= 0.0 || tick_value <= 0.0)
      return 0.0;
   double target_usd = FINTOKEI_COMMISSION_RT_USD * lot + FINTOKEI_MIN_NET_PROFIT_USD;
   return target_usd * tick_size / (tick_value * lot);
}

//+------------------------------------------------------------------+
double CspaExit_BreakevenSl(const CspaExitTrack &t, const double trail_atr)
{
   double atr_buffer = t.be_buffer_atr * trail_atr;
   double comm_buffer = CspaExit_CommissionSlBuffer(t.symbol, t.lot_size);
   double buffer = MathMax(atr_buffer, comm_buffer);
   if(t.direction == POSITION_TYPE_BUY)
      return t.entry + buffer;
   return t.entry - buffer;
}

//+------------------------------------------------------------------+
double CspaExit_NormalizePrice(const string symbol, const double price)
{
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   return NormalizeDouble(price, digits);
}

//+------------------------------------------------------------------+
bool CspaExit_ModifySl(
   const ulong ticket,
   const string symbol,
   const double new_sl,
   const double tp
)
{
   if(!PositionSelectByTicket(ticket))
      return false;

   double cur_sl = PositionGetDouble(POSITION_SL);
   double point  = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point <= 0.0)
      point = 0.00001;
   if(MathAbs(new_sl - cur_sl) < point * 0.5)
      return true;

   double norm_sl = CspaExit_NormalizePrice(symbol, new_sl);
   double norm_tp = CspaExit_NormalizePrice(symbol, tp);

   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action   = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.symbol   = symbol;
   request.sl       = norm_sl;
   request.tp       = norm_tp;
   request.magic    = PositionGetInteger(POSITION_MAGIC);

   if(!OrderSend(request, result))
   {
      Print("CspaExit ModifySL failed ticket=", ticket,
            " retcode=", result.retcode, " ", result.comment);
      return false;
   }
   if(result.retcode != TRADE_RETCODE_DONE)
   {
      Print("CspaExit ModifySL retcode=", result.retcode, " ", result.comment);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
void CspaExit_ProcessBar(
   const int idx,
   const double high,
   const double low,
   const double close
)
{
   CspaExitTrack t = g_cspa_tracks[idx];
   if(!t.active)
      return;

   double risk = CspaExit_InitialRisk(t);
   if(risk <= 0.0)
      return;

   g_cspa_tracks[idx].bars_since_entry++;

   double trail_atr = MathMax(t.atr, risk * 0.25);
   double be_sl     = CspaExit_BreakevenSl(t, trail_atr);
   double bar_mfe_r = 0.0;

   if(t.direction == POSITION_TYPE_BUY)
   {
      bar_mfe_r = (high - t.entry) / risk;
      g_cspa_tracks[idx].peak_favorable = MathMax(t.peak_favorable, high);
   }
   else
   {
      bar_mfe_r = (t.entry - low) / risk;
      g_cspa_tracks[idx].peak_favorable = MathMin(t.peak_favorable, low);
   }

   t = g_cspa_tracks[idx];

   if(t.be_enabled)
   {
      if(bar_mfe_r >= t.be_arm_mfe_r)
         g_cspa_tracks[idx].extension_armed = true;

      double close_r = CspaExit_ProfitR(t, close);
      bool rhythm_window = (t.bars_since_entry <= t.be_rhythm_max_bars);

      if(bar_mfe_r >= t.be_trigger_mfe_r)
      {
         g_cspa_tracks[idx].current_sl = CspaExit_RatchetSl(
            t, t.current_sl, be_sl
         );
         g_cspa_tracks[idx].sl_at_breakeven = true;
      }
      else if(t.extension_armed && rhythm_window && close_r <= t.be_pullback_close_r)
      {
         g_cspa_tracks[idx].current_sl = CspaExit_RatchetSl(
            t, t.current_sl, be_sl
         );
         g_cspa_tracks[idx].sl_at_breakeven = true;
      }
   }

   t = g_cspa_tracks[idx];

   if(t.trail_enabled && t.sl_at_breakeven)
   {
      double trail_sl = 0.0;
      if(t.direction == POSITION_TYPE_BUY)
      {
         trail_sl = t.peak_favorable - t.trail_atr_mult * trail_atr;
         trail_sl = MathMax(trail_sl, be_sl);
      }
      else
      {
         trail_sl = t.peak_favorable + t.trail_atr_mult * trail_atr;
         trail_sl = MathMin(trail_sl, be_sl);
      }
      g_cspa_tracks[idx].current_sl = CspaExit_RatchetSl(
         t, t.current_sl, trail_sl
      );
   }

   t = g_cspa_tracks[idx];
   CspaExit_ModifySl(t.ticket, t.symbol, t.current_sl, t.take_profit);
}

//+------------------------------------------------------------------+
void CspaExit_Register(
   const ulong ticket,
   const string symbol,
   const long direction,
   const double entry,
   const double initial_sl,
   const double take_profit,
   const double atr,
   const bool be_enabled,
   const bool trail_enabled,
   const double be_arm_mfe_r,
   const double be_trigger_mfe_r,
   const double be_pullback_close_r,
   const int be_rhythm_max_bars,
   const double trail_atr_mult,
   const double be_buffer_atr
)
{
   int idx = CspaExit_FindByTicket(ticket);
   if(idx < 0)
   {
      idx = CspaExit_FindSlot();
      if(idx < 0)
      {
         Print("CspaExit: track table full, cannot register ticket=", ticket);
         return;
      }
   }

   g_cspa_tracks[idx].active              = true;
   g_cspa_tracks[idx].ticket            = ticket;
   g_cspa_tracks[idx].symbol            = symbol;
   g_cspa_tracks[idx].direction         = direction;
   g_cspa_tracks[idx].entry             = entry;
   g_cspa_tracks[idx].initial_sl        = initial_sl;
   g_cspa_tracks[idx].take_profit       = take_profit;
   g_cspa_tracks[idx].atr               = atr;
   g_cspa_tracks[idx].lot_size          = 0.0;
   if(PositionSelectByTicket(ticket))
      g_cspa_tracks[idx].lot_size = PositionGetDouble(POSITION_VOLUME);
   g_cspa_tracks[idx].be_enabled        = be_enabled;
   g_cspa_tracks[idx].trail_enabled     = trail_enabled;
   g_cspa_tracks[idx].be_arm_mfe_r      = be_arm_mfe_r;
   g_cspa_tracks[idx].be_trigger_mfe_r = be_trigger_mfe_r;
   g_cspa_tracks[idx].be_pullback_close_r = be_pullback_close_r;
   g_cspa_tracks[idx].be_rhythm_max_bars  = be_rhythm_max_bars;
   g_cspa_tracks[idx].trail_atr_mult    = trail_atr_mult;
   g_cspa_tracks[idx].be_buffer_atr     = be_buffer_atr;
   g_cspa_tracks[idx].current_sl        = initial_sl;
   g_cspa_tracks[idx].peak_favorable    = entry;
   g_cspa_tracks[idx].extension_armed   = false;
   g_cspa_tracks[idx].sl_at_breakeven   = false;
   g_cspa_tracks[idx].bars_since_entry  = 0;
   g_cspa_tracks[idx].last_bar_time     = 0;

   Print(
      "CspaExit registered ticket=", ticket,
      " entry=", DoubleToString(entry, 5),
      " sl=", DoubleToString(initial_sl, 5),
      " atr=", DoubleToString(atr, 5)
   );
}

//+------------------------------------------------------------------+
bool CspaExit_ExtractBool(const string json, const string key, const bool default_val)
{
   string pattern = "\"" + key + "\":";
   int pos = StringFind(json, pattern);
   if(pos < 0)
      return default_val;
   pos += StringLen(pattern);
   while(pos < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, pos);
      if(ch != ' ' && ch != '\t')
         break;
      pos++;
   }
   string tail = StringSubstr(json, pos, 4);
   if(StringFind(tail, "true") == 0 || StringFind(tail, "1") == 0)
      return true;
   if(StringFind(tail, "false") == 0 || StringFind(tail, "0") == 0)
      return false;
   return default_val;
}

//+------------------------------------------------------------------+
bool CspaExit_ExtractInt(const string json, const string key, int &value)
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
void CspaExit_TryRegisterFromSignal(
   const ulong ticket,
   const string symbol,
   const long direction,
   const double entry,
   const double initial_sl,
   const double take_profit,
   const string response_json
)
{
   string setup_type, exit_mode;
   if(!ExtractJsonString(response_json, "setup_type", setup_type))
      return;
   if(setup_type != "CSPA")
      return;
   if(!ExtractJsonString(response_json, "exit_mode", exit_mode))
      return;
   if(exit_mode != "CSPA_BE_TRAIL")
      return;

   double atr = 0.0;
   double be_arm = 0.35;
   double be_trigger = 0.5;
   double be_pullback = 0.08;
   double trail_mult = 1.0;
   double be_buffer = 0.0;
   int rhythm_bars = 12;
   int be_enabled_i = 1;
   int trail_enabled_i = 1;

   ExtractJsonDouble(response_json, "exit_atr", atr);
   CspaExit_ExtractInt(response_json, "exit_be_enabled", be_enabled_i);
   CspaExit_ExtractInt(response_json, "exit_trail_enabled", trail_enabled_i);
   ExtractJsonDouble(response_json, "exit_be_arm_mfe_r", be_arm);
   ExtractJsonDouble(response_json, "exit_be_trigger_mfe_r", be_trigger);
   ExtractJsonDouble(response_json, "exit_be_pullback_close_r", be_pullback);
   CspaExit_ExtractInt(response_json, "exit_be_rhythm_max_bars", rhythm_bars);
   ExtractJsonDouble(response_json, "exit_trail_atr_mult", trail_mult);
   ExtractJsonDouble(response_json, "exit_be_buffer_atr", be_buffer);

   CspaExit_Register(
      ticket,
      symbol,
      direction,
      entry,
      initial_sl,
      take_profit,
      atr,
      be_enabled_i != 0,
      trail_enabled_i != 0,
      be_arm,
      be_trigger,
      be_pullback,
      rhythm_bars,
      trail_mult,
      be_buffer
   );
}

//+------------------------------------------------------------------+
void CspaExit_PurgeClosed()
{
   for(int i = 0; i < CSPA_EXIT_MAX_TRACKS; i++)
   {
      if(!g_cspa_tracks[i].active)
         continue;
      if(!PositionSelectByTicket(g_cspa_tracks[i].ticket))
         CspaExit_ClearSlot(i);
   }
}

//+------------------------------------------------------------------+
void CspaExit_PurgeByPosition(const ulong position_id)
{
   if(position_id == 0)
      return;
   for(int i = 0; i < CSPA_EXIT_MAX_TRACKS; i++)
   {
      if(!g_cspa_tracks[i].active)
         continue;
      if(g_cspa_tracks[i].ticket == position_id)
         CspaExit_ClearSlot(i);
   }
}

//+------------------------------------------------------------------+
void CspaExit_OnNewBar(const string symbol, const ENUM_TIMEFRAMES tf)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(symbol, tf, 1, 1, rates) != 1)
      return;

   datetime bar_time = rates[0].time;
   double high  = rates[0].high;
   double low   = rates[0].low;
   double close = rates[0].close;

   for(int i = 0; i < CSPA_EXIT_MAX_TRACKS; i++)
   {
      if(!g_cspa_tracks[i].active)
         continue;
      if(g_cspa_tracks[i].symbol != symbol)
         continue;
      if(g_cspa_tracks[i].last_bar_time == bar_time)
         continue;

      g_cspa_tracks[i].last_bar_time = bar_time;
      CspaExit_ProcessBar(i, high, low, close);
   }
}

//+------------------------------------------------------------------+
void CspaExit_ManageOpenPositions(const ulong magic, const ENUM_TIMEFRAMES tf)
{
   CspaExit_PurgeClosed();

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
      CspaExit_OnNewBar(symbol, tf);
   }
}

#endif // CSPA_EXIT_MANAGER_MQH
