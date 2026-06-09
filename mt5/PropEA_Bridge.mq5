//+------------------------------------------------------------------+
//| PropEA_Bridge.mq5 — 7層多層防御EA MT5 ↔ Python HTTP ブリッジ雛形   |
//| Python: uvicorn mt5_bridge:app --host 127.0.0.1 --port 8000       |
//| MT5: ツール→オプション→エキスパートアドバイザー→WebRequest許可URL |
//+------------------------------------------------------------------+
#property copyright "Prop EA Project"
#property version   "1.00"

input string InpApiUrl              = "http://127.0.0.1:8000/trade_signal";
input int    InpTimerSeconds        = 60;          // シグナル照会間隔（秒）
input int    InpHistoryBars         = 300;         // Pythonへ送るM5履歴本数
input int    InpMinutesToNews       = 45;          // 次ニュースまでの分数（暫定）
input string InpNewsImpact          = "HIGH";      // LOW / MEDIUM / HIGH
input double InpMaxSpreadPoints     = 30;          // 最大許容スプレッド
input ulong  InpMagic               = 20260601;
input bool   InpSendCorrelatedPair  = true;        // SMT/CORRELATION用に相関ペアも送信

input bool   InpPyramidLiveEnabled    = true;        // Live Limit ピラミッド (Python /pyramid/*)

#include "LiveSentinel.mqh"
#include "CspaExitManager.mqh"
#include "PyramidLiveManager.mqh"

string g_correlated_symbol = "";
datetime g_last_bar_time   = 0;
datetime g_last_request    = 0;

//+------------------------------------------------------------------+
string CanonicalPair(const string symbol)
{
   string upper = symbol;
   StringToUpper(upper);
   StringReplace(upper, ".", "");
   StringReplace(upper, "_", "");
   StringReplace(upper, "-", "");
   if(StringFind(upper, "GBPUSD") >= 0) return "GBPUSD";
   if(StringFind(upper, "EURUSD") >= 0) return "EURUSD";
   return upper;
}

//+------------------------------------------------------------------+
string ExtractPairSuffix(const string symbol, const string canonical)
{
   string upper = symbol;
   StringToUpper(upper);
   StringReplace(upper, ".", "");
   int pos = StringFind(upper, canonical);
   if(pos < 0) return "";
   return StringSubstr(symbol, pos + StringLen(canonical));
}

//+------------------------------------------------------------------+
bool SymbolExists(const string symbol)
{
   return (bool)SymbolInfoInteger(symbol, SYMBOL_EXIST);
}

//+------------------------------------------------------------------+
string ResolveBrokerSymbol(const string canonical, const string reference_symbol)
{
   string suffix = ExtractPairSuffix(reference_symbol, CanonicalPair(reference_symbol));
   string candidate = canonical + suffix;
   if(SymbolExists(candidate))
      return candidate;

   // サフィックス付き候補を総当たり（Fintokei: EURUSDp 等）
   for(int i = 0; i < SymbolsTotal(false); i++)
   {
      string sym = SymbolName(i, false);
      if(CanonicalPair(sym) == canonical)
         return sym;
   }
   return candidate;
}

//+------------------------------------------------------------------+
string CorrelatedSymbol(const string symbol)
{
   string canonical = CanonicalPair(symbol);
   if(canonical == "GBPUSD") return ResolveBrokerSymbol("EURUSD", symbol);
   if(canonical == "EURUSD") return ResolveBrokerSymbol("GBPUSD", symbol);
   return "";
}

//+------------------------------------------------------------------+
string JsonEscape(const string text)
{
   string out = text;
   StringReplace(out, "\\", "\\\\");
   StringReplace(out, "\"", "\\\"");
   return out;
}

//+------------------------------------------------------------------+
string FormatBarTime(const datetime t)
{
   MqlDateTime dt;
   TimeToStruct(t, dt);
   return StringFormat("%04d-%02d-%02d %02d:%02d:%02d", dt.year, dt.mon, dt.day, dt.hour, dt.min, dt.sec);
}

//+------------------------------------------------------------------+
string BuildBarsJson(const string symbol, const ENUM_TIMEFRAMES tf, const int count)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(symbol, tf, 0, count, rates);
   if(copied <= 0)
      return "";

   string json = "[";
   for(int i = copied - 1; i >= 0; i--)
   {
      if(i != copied - 1) json += ",";
      json += StringFormat(
         "{\"time\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%.0f}",
         FormatBarTime(rates[i].time),
         rates[i].open,
         rates[i].high,
         rates[i].low,
         rates[i].close,
         (double)rates[i].tick_volume
      );
   }
   json += "]";
   return json;
}

//+------------------------------------------------------------------+
string BuildMarketBlock(const string symbol, const ENUM_TIMEFRAMES tf)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(symbol, tf, 0, 1, rates) != 1)
      return "";

   string pair = CanonicalPair(symbol);

   return StringFormat(
      "\"market\":{\"pair\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%.0f}",
      JsonEscape(pair),
      rates[0].open,
      rates[0].high,
      rates[0].low,
      rates[0].close,
      (double)rates[0].tick_volume
   );
}

//+------------------------------------------------------------------+
string BuildRequestJson(const string symbol, const ENUM_TIMEFRAMES tf)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(symbol, tf, 0, 1, rates) != 1)
      return "";

   string pair = CanonicalPair(symbol);

   string bars_json = BuildBarsJson(symbol, tf, InpHistoryBars);
   if(bars_json == "")
      return "";

   string json = "{";
   json += BuildMarketBlock(symbol, tf);
   json += StringFormat(
      ",\"calendar\":{\"minutes_to_next_news\":%d,\"news_impact_level\":\"%s\"}",
      InpMinutesToNews,
      JsonEscape(InpNewsImpact)
   );
   long spread_pts = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   json += StringFormat(
      ",\"account\":{\"equity\":%.2f,\"balance\":%.2f}",
      AccountInfoDouble(ACCOUNT_EQUITY),
      AccountInfoDouble(ACCOUNT_BALANCE)
   );
   json += StringFormat(",\"bar_time\":\"%s\"", FormatBarTime(rates[0].time));
   json += StringFormat(",\"server_time\":\"%s\"", FormatBarTime(TimeCurrent()));
   json += StringFormat(",\"spread_points\":%d", (int)spread_pts);
   json += StringFormat(",\"bars\":%s", bars_json);

   if(InpSendCorrelatedPair && g_correlated_symbol != "")
   {
      MqlRates corr_rates[];
      ArraySetAsSeries(corr_rates, true);
      string corr_bars = BuildBarsJson(g_correlated_symbol, tf, InpHistoryBars);
      if(CopyRates(g_correlated_symbol, tf, 0, 1, corr_rates) == 1 && corr_bars != "")
      {
         string corr_pair = CanonicalPair(g_correlated_symbol);
         json += StringFormat(
            ",\"correlated_market\":{\"pair\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%.0f}",
            JsonEscape(corr_pair),
            corr_rates[0].open,
            corr_rates[0].high,
            corr_rates[0].low,
            corr_rates[0].close,
            (double)corr_rates[0].tick_volume
         );
         json += StringFormat(",\"correlated_bar_time\":\"%s\"", FormatBarTime(corr_rates[0].time));
         json += StringFormat(",\"correlated_bars\":%s", corr_bars);
      }
   }

   json += "}";
   return json;
}

//+------------------------------------------------------------------+
bool ExtractJsonString(const string json, const string key, string &value)
{
   string pattern = "\"" + key + "\":\"";
   int pos = StringFind(json, pattern);
   if(pos < 0) return false;
   pos += StringLen(pattern);
   int end = StringFind(json, "\"", pos);
   if(end < 0) return false;
   value = StringSubstr(json, pos, end - pos);
   return true;
}

//+------------------------------------------------------------------+
bool ExtractJsonDouble(const string json, const string key, double &value)
{
   string pattern = "\"" + key + "\":";
   int pos = StringFind(json, pattern);
   if(pos < 0) return false;
   pos += StringLen(pattern);
   int end = pos;
   while(end < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, end);
      if(ch == ',' || ch == '}' || ch == ']') break;
      end++;
   }
   value = StringToDouble(StringSubstr(json, pos, end - pos));
   return true;
}

//+------------------------------------------------------------------+
bool PostTradeSignal(const string body, string &response)
{
   char post[];
   char result[];
   string result_headers;
   StringToCharArray(body, post, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayResize(post, StringLen(body));

   string headers = "Content-Type: application/json\r\n";
   int timeout_ms = 5000;
   int status = WebRequest(
      "POST",
      InpApiUrl,
      headers,
      timeout_ms,
      post,
      result,
      result_headers
   );

   if(status == -1)
   {
      Print("WebRequest failed. err=", GetLastError(),
            " — 許可URLとPythonサーバー起動を確認してください。");
      return false;
   }
   if(status != 200)
   {
      Print("HTTP status=", status, " body=", CharArrayToString(result));
      return false;
   }

   response = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
   return true;
}

//+------------------------------------------------------------------+
bool HasOpenPosition(const string symbol)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) == symbol &&
         PositionGetInteger(POSITION_MAGIC) == (long)InpMagic)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING ResolveFillingMode(const string symbol)
{
   int filling = (int)SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      return ORDER_FILLING_FOK;
   if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      return ORDER_FILLING_IOC;
   return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
double NormalizeLot(const string symbol, const double lot)
{
   double min_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double step    = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0.0)
      step = 0.01;
   double out = MathMax(min_lot, MathMin(max_lot, lot));
   out = MathFloor(out / step) * step;
   return out;
}

//+------------------------------------------------------------------+
double CalcLotFromRiskBudget(
   const string symbol,
   const double risk_budget,
   const double entry,
   const double sl
)
{
   // L4.5: lot = risk_budget / (SL距離[ticks] × tick_value)
   if(risk_budget <= 0.0)
      return 0.0;
   double sl_distance = MathAbs(entry - sl);
   if(sl_distance <= 0.0)
      return 0.0;

   double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tick_size <= 0.0 || tick_value <= 0.0)
      return 0.0;

   double loss_per_lot = (sl_distance / tick_size) * tick_value;
   if(loss_per_lot <= 0.0)
      return 0.0;
   return NormalizeLot(symbol, risk_budget / loss_per_lot);
}

//+------------------------------------------------------------------+
bool execute_trade(
   const string symbol,
   const string action,
   const double risk_budget,
   const double entry,
   const double sl,
   const double tp,
   const double lot_fallback,
   const string comment,
   const string response_json
)
{
   if(action != "BUY" && action != "SELL")
      return false;

   if(!LiveSentinel_EntryAllowed(TimeCurrent(), SymbolInfoInteger(symbol, SYMBOL_SPREAD), InpMaxSpreadPoints))
   {
      Print("execute_trade skip - Live Sentinel entry block");
      return false;
   }

   if(HasOpenPosition(symbol))
   {
      Print("execute_trade skip - position already open: ", symbol);
      return false;
   }

   long spread = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   if(spread > InpMaxSpreadPoints)
   {
      Print("execute_trade skip - spread too wide: ", spread);
      return false;
   }

   double lot = CalcLotFromRiskBudget(symbol, risk_budget, entry, sl);
   if(lot <= 0.0 && lot_fallback > 0.0)
      lot = NormalizeLot(symbol, lot_fallback);
   if(lot <= 0.0)
   {
      Print("execute_trade skip - lot is zero (risk_budget=", risk_budget, ")");
      return false;
   }

   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action       = TRADE_ACTION_DEAL;
   request.symbol       = symbol;
   request.volume       = lot;
   request.type         = (action == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   request.price        = (action == "BUY")
      ? SymbolInfoDouble(symbol, SYMBOL_ASK)
      : SymbolInfoDouble(symbol, SYMBOL_BID);
   request.sl           = sl;
   request.tp           = tp;
   request.deviation    = 20;
   request.magic        = InpMagic;
   request.comment      = comment;
   request.type_filling = ResolveFillingMode(symbol);

   if(!OrderSend(request, result))
   {
      Print("execute_trade OrderSend failed: retcode=", result.retcode, " ", result.comment);
      return false;
   }

   Print(
      "execute_trade OK action=", action,
      " lot=", lot,
      " risk_budget=", risk_budget,
      " ticket=", result.order
   );

   ulong pos_ticket = 0;
   for(int pi = PositionsTotal() - 1; pi >= 0; pi--)
   {
      ulong t = PositionGetTicket(pi);
      if(t == 0)
         continue;
      if(!PositionSelectByTicket(t))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagic)
         continue;
      pos_ticket = t;
      break;
   }

   if(pos_ticket > 0)
   {
      long pos_dir = PositionGetInteger(POSITION_TYPE);
      double fill_entry = PositionGetDouble(POSITION_PRICE_OPEN);
      CspaExit_TryRegisterFromSignal(
         pos_ticket,
         symbol,
         pos_dir,
         fill_entry,
         sl,
         tp,
         response_json
      );

      if(InpPyramidLiveEnabled)
      {
         PyramidLive_RegisterAfterEntry(
            symbol,
            PERIOD_M5,
            response_json,
            pos_ticket,
            fill_entry,
            sl,
            tp,
            lot,
            InpMagic
         );
      }
   }

   return true;
}

//+------------------------------------------------------------------+
bool close_trade(const string symbol)
{
   if(LiveSentinel_ShouldHoldLogicClose(symbol, InpMaxSpreadPoints))
      return false;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagic)
         continue;

      long pos_type = PositionGetInteger(POSITION_TYPE);
      double volume = PositionGetDouble(POSITION_VOLUME);

      MqlTradeRequest request;
      MqlTradeResult  result;
      ZeroMemory(request);
      ZeroMemory(result);

      request.action       = TRADE_ACTION_DEAL;
      request.symbol       = symbol;
      request.volume       = volume;
      request.position     = ticket;
      request.deviation    = 20;
      request.magic        = InpMagic;
      request.comment      = "PropEA_close";
      request.type_filling = ResolveFillingMode(symbol);

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
         Print("close_trade failed: retcode=", result.retcode, " ", result.comment);
         return false;
      }

      Print("close_trade OK ticket=", ticket, " volume=", volume);
      return true;
   }

   Print("close_trade - no open position for ", symbol);
   return false;
}

//+------------------------------------------------------------------+
void ExecuteSignal(const string symbol, const string response_json)
{
   string action, message;
   double lot_size = 0, sl = 0, tp = 0, entry = 0, risk_budget = 0;

   if(!ExtractJsonString(response_json, "action", action))
   {
      Print("Parse error: action missing");
      return;
   }
   ExtractJsonDouble(response_json, "lot_size", lot_size);
   ExtractJsonDouble(response_json, "risk_budget", risk_budget);
   ExtractJsonDouble(response_json, "sl", sl);
   ExtractJsonDouble(response_json, "tp", tp);
   ExtractJsonDouble(response_json, "entry", entry);
   ExtractJsonString(response_json, "message", message);

   Print(
      "Python signal: action=", action,
      " risk_budget=", risk_budget,
      " lot=", lot_size,
      " entry=", entry,
      " sl=", sl,
      " tp=", tp,
      " | ", message
   );

   if(action == "PANIC_CLOSE")
   {
      Print("LIVE_SENTINEL Python PANIC_CLOSE: ", message);
      LiveSentinel_PanicCloseAll(InpMagic);
      LiveSentinel_CancelAllPending(InpMagic);
      g_ls_entry_locked = true;
      g_ls_terminator_fired = true;
      return;
   }

   if(action == "HOLD" || action == "REJECT")
      return;

   if(entry <= 0.0)
      entry = (action == "BUY")
         ? SymbolInfoDouble(symbol, SYMBOL_ASK)
         : SymbolInfoDouble(symbol, SYMBOL_BID);

   if(!execute_trade(symbol, action, risk_budget, entry, sl, tp, lot_size, message, response_json))
      Print("execute_trade failed for action=", action);
}

//+------------------------------------------------------------------+
void RequestPipelineSignal()
{
   string symbol = _Symbol;
   ENUM_TIMEFRAMES tf = PERIOD_M5;

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(symbol, tf, 0, 1, rates) != 1)
      return;

   datetime server_time = TimeCurrent();
   long spread_pts = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   if(!LiveSentinel_EntryAllowed(server_time, spread_pts, InpMaxSpreadPoints))
      return;

   // 新バー確定時のみ照会（OnTimerと併用）
   if(rates[0].time == g_last_bar_time && TimeCurrent() - g_last_request < InpTimerSeconds)
      return;

   string body = BuildRequestJson(symbol, tf);
   if(body == "")
   {
      Print("Failed to build JSON payload");
      return;
   }

   string response;
   if(!PostTradeSignal(body, response))
      return;

   g_last_bar_time = rates[0].time;
   g_last_request  = TimeCurrent();
   ExecuteSignal(symbol, response);
}

//+------------------------------------------------------------------+
int OnInit()
{
   g_correlated_symbol = CorrelatedSymbol(_Symbol);
   if(InpSendCorrelatedPair && g_correlated_symbol != "")
      SymbolSelect(g_correlated_symbol, true);

   PyramidLive_SetApiBaseFromTradeUrl(InpApiUrl);
   EventSetTimer(InpTimerSeconds);
   Print("PropEA_Bridge initialized. API=", InpApiUrl, " corr=", g_correlated_symbol,
         " pyramid_live=", (InpPyramidLiveEnabled ? "on" : "off"));
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
}

//+------------------------------------------------------------------+
void OnTimer()
{
   RequestPipelineSignal();
}

//+------------------------------------------------------------------+
void OnTick()
{
   LiveSentinel_OnTick(InpMagic);
   CspaExit_ManageOpenPositions(InpMagic, PERIOD_M5);

   if(InpPyramidLiveEnabled)
   {
      PyramidLive_OnNewBar(_Symbol, PERIOD_M5, InpMagic, 5.0);
      PyramidLive_PruneClosedTracks(_Symbol, InpMagic);
   }

   // タイマー間隔より短い周期で新バー検知したい場合の補助
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(_Symbol, PERIOD_M5, 0, 1, rates) == 1 && rates[0].time != g_last_bar_time)
      RequestPipelineSignal();
}

//+------------------------------------------------------------------+
void OnTradeTransaction(
   const MqlTradeTransaction &trans,
   const MqlTradeRequest &request,
   const MqlTradeResult &result
)
{
   if(InpPyramidLiveEnabled)
      PyramidLive_OnTradeTransaction(trans, request, result);
}

//+------------------------------------------------------------------+
