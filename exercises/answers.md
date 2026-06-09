# SOLUTIONS

TÃ i liá»‡u nÃ y tráº£ lá»i ngáº¯n gá»n cÃ¡c cÃ¢u há»i trong `CODELAB.md` vÃ  `exercises/README.md`.

## Pháº§n 1: Direct LLM Calling

### 1. LLM Ä‘Æ°á»£c khá»Ÿi táº¡o nhÆ° tháº¿ nÃ o?
LLM Ä‘Æ°á»£c khá»Ÿi táº¡o qua hÃ m `get_llm()` trong `common/llm.py`. HÃ m nÃ y táº¡o `ChatOpenAI` vÃ  trá» tá»›i OpenRouter.

### 2. Message Ä‘Æ°á»£c gá»­i Ä‘áº¿n LLM cÃ³ cáº¥u trÃºc gÃ¬?
Danh sÃ¡ch message gá»“m Ã­t nháº¥t `SystemMessage` Ä‘á»ƒ Ä‘áº·t vai trÃ²/hÃ nh vi cho model vÃ  `HumanMessage` Ä‘á»ƒ chá»©a cÃ¢u há»i cá»§a ngÆ°á»i dÃ¹ng.

### 3. VÃ¬ sao cáº§n `SystemMessage` vÃ  `HumanMessage`?
`SystemMessage` Ä‘iá»u khiá»ƒn cÃ¡ch model tráº£ lá»i, cÃ²n `HumanMessage` lÃ  Ä‘áº§u vÃ o thá»±c táº¿ cáº§n xá»­ lÃ½. TÃ¡ch hai loáº¡i message giÃºp prompt rÃµ rÃ ng vÃ  á»•n Ä‘á»‹nh hÆ¡n.

## Pháº§n 2: LLM + RAG & Tools

### 1. `@tool` decorator Ä‘Æ°á»£c dÃ¹ng á»Ÿ Ä‘Ã¢u?
Trong `stages/stage_2_rag_tools/main.py`, cÃ¡c hÃ m nhÆ° `search_legal_database` vÃ  `calculate_penalty` Ä‘Æ°á»£c gáº¯n `@tool` Ä‘á»ƒ LLM cÃ³ thá»ƒ gá»i chÃºng.

### 2. `LEGAL_KNOWLEDGE` Ä‘Æ°á»£c cáº¥u trÃºc nhÆ° tháº¿ nÃ o?
ÄÃ³ lÃ  má»™t danh sÃ¡ch cÃ¡c dictionary, má»—i pháº§n tá»­ cÃ³ `id`, `keywords`, vÃ  `text`. `keywords` dÃ¹ng Ä‘á»ƒ dÃ² khá»›p truy váº¥n, `text` lÃ  ná»™i dung phÃ¡p lÃ½ tráº£ vá».

### 3. LLM Ä‘Æ°á»£c bind vá»›i tools ra sao?
DÃ¹ng `llm.bind_tools(tools)`. Khi Ä‘Ã³ model cÃ³ thá»ƒ sinh ra `tool_calls`, rá»“i chÆ°Æ¡ng trÃ¬nh tá»± thá»±c thi tool vÃ  Ä‘Æ°a káº¿t quáº£ ngÆ°á»£c láº¡i cho LLM.

## Pháº§n 3: Single Agent vá»›i ReAct

### 1. `create_react_agent()` lÃ  gÃ¬?
ÄÃ¢y lÃ  hÃ m dá»±ng sáºµn cá»§a LangGraph Ä‘á»ƒ táº¡o agent theo chu trÃ¬nh ReAct: suy nghÄ©, gá»i tool, quan sÃ¡t, rá»“i láº·p láº¡i cho Ä‘áº¿n khi cÃ³ cÃ¢u tráº£ lá»i cuá»‘i.

### 2. KhÃ¡c gÃ¬ so vá»›i Stage 2?
Stage 2 pháº£i tá»± viáº¿t vÃ²ng láº·p gá»i tool. Stage 3 Ä‘á»ƒ agent tá»± quyáº¿t Ä‘á»‹nh lÃºc nÃ o cáº§n gá»i tool vÃ  gá»i bao nhiÃªu láº§n.

### 3. `agent_executor.invoke()` hay `agent.ainvoke()` chá»‰ cáº§n gá»i má»™t láº§n vÃ¬ sao?
VÃ¬ ReAct agent tá»± quáº£n lÃ½ toÃ n bá»™ vÃ²ng suy luáº­n vÃ  tool calling bÃªn trong. NgÆ°á»i gá»i chá»‰ cáº§n truyá»n cÃ¢u há»i ban Ä‘áº§u.

## Pháº§n 4: Multi-Agent In-Process

### 1. `class State(TypedDict)` dÃ¹ng Ä‘á»ƒ lÃ m gÃ¬?
NÃ³ Ä‘á»‹nh nghÄ©a state dÃ¹ng chung giá»¯a cÃ¡c node trong graph. Má»—i node Ä‘á»c vÃ  ghi má»™t pháº§n cá»§a state.

### 2. CÃ¡c agent function lÃ m gÃ¬?
`law_agent` phÃ¢n tÃ­ch phÃ¡p lÃ½ tá»•ng quÃ¡t, `tax_agent` xá»­ lÃ½ khÃ­a cáº¡nh thuáº¿, `compliance_agent` xá»­ lÃ½ tuÃ¢n thá»§, vÃ  `aggregate_results` tá»•ng há»£p Ä‘áº§u ra.

### 3. `Send()` API dÃ¹ng Ä‘á»ƒ lÃ m gÃ¬?
`Send()` dÃ¹ng Ä‘á»ƒ gá»­i nhiá»u nhÃ¡nh xá»­ lÃ½ song song tá»« má»™t node sang cÃ¡c node khÃ¡c trong LangGraph.

### 4. `graph.add_node()` vÃ  `graph.add_edge()` khÃ¡c nhau tháº¿ nÃ o?
`add_node()` Ä‘Äƒng kÃ½ má»™t node xá»­ lÃ½, cÃ²n `add_edge()` mÃ´ táº£ luá»“ng Ä‘i giá»¯a cÃ¡c node.

### 5. `privacy_agent` nÃªn lÃ m gÃ¬?
NÃ³ phÃ¢n tÃ­ch cÃ¡c váº¥n Ä‘á» vá» GDPR, data protection, privacy rights, data breach vÃ  dá»¯ liá»‡u cÃ¡ nhÃ¢n khi cÃ¢u há»i cÃ³ liÃªn quan.

### 6. Conditional routing nÃªn hoáº¡t Ä‘á»™ng tháº¿ nÃ o?
`check_routing()` kiá»ƒm tra tá»« khÃ³a trong cÃ¢u há»i. Náº¿u cÃ³ `data`, `privacy`, `gdpr`, hoáº·c `dá»¯ liá»‡u` thÃ¬ má»›i gá»­i thÃªm nhÃ¡nh `privacy_agent`.

## Pháº§n 5: Distributed A2A System

### 1. Trace request flow lÃ  gÃ¬?
LÃ  theo dÃµi Ä‘Æ°á»ng Ä‘i cá»§a má»™t request qua `Customer Agent -> Law Agent -> Tax/Compliance Agent` thÃ´ng qua `trace_id` hoáº·c log.

### 2. Test dynamic discovery Ä‘á»ƒ lÃ m gÃ¬?
Äá»ƒ kiá»ƒm tra Registry cÃ³ tá»± tÃ¬m Ä‘Æ°á»£c agent Ä‘ang hoáº¡t Ä‘á»™ng hay khÃ´ng. Náº¿u má»™t agent dá»«ng, há»‡ thá»‘ng pháº£i pháº£n á»©ng Ä‘Ãºng thay vÃ¬ phá»¥ thuá»™c URL hardcode.

### 3. Náº¿u sá»­a prompt cá»§a `tax_agent/graph.py` thÃ¬ tÃ¡c Ä‘á»™ng gÃ¬?
Agent sáº½ thay Ä‘á»•i Ä‘á»™ dÃ i, phong cÃ¡ch vÃ  Ä‘á»™ chi tiáº¿t cá»§a cÃ¢u tráº£ lá»i. Sau khi sá»­a cáº§n restart agent Ä‘á»ƒ prompt má»›i cÃ³ hiá»‡u lá»±c.

## Pháº§n 6: CÃ¢u Há»i Ã”n Táº­p

### 1. Khi nÃ o nÃªn dÃ¹ng single agent thay vÃ¬ multi-agent?
DÃ¹ng single agent khi bÃ i toÃ¡n nhá», Ã­t miá»n kiáº¿n thá»©c, luá»“ng xá»­ lÃ½ Ä‘Æ¡n giáº£n, vÃ  khÃ´ng cáº§n song song hÃ³a.

### 2. Æ¯u Ä‘iá»ƒm cá»§a A2A protocol so vá»›i gRPC hoáº·c REST thÃ´ng thÆ°á»ng?
A2A chuáº©n hÃ³a giao tiáº¿p giá»¯a agent, há»— trá»£ agent card, discovery, vÃ  message semantics phÃ¹ há»£p hÆ¡n cho há»‡ nhiá»u agent.

### 3. LÃ m tháº¿ nÃ o Ä‘á»ƒ ngÄƒn infinite delegation loops trong A2A?
Äáº·t giá»›i háº¡n Ä‘á»™ sÃ¢u, theo dÃµi trace/context, vÃ  dá»«ng khi Ä‘Ã£ vÆ°á»£t quÃ¡ sá»‘ hop cho phÃ©p.

### 4. Táº¡i sao cáº§n Registry service?
Registry cho phÃ©p agent tá»± Ä‘Äƒng kÃ½ vÃ  Ä‘Æ°á»£c khÃ¡m phÃ¡ Ä‘á»™ng lÃºc runtime, trÃ¡nh hardcode URL vÃ  giÃºp thay Ä‘á»•i/scale linh hoáº¡t hÆ¡n.

## BÃ i Táº­p Cá»™ng Äiá»ƒm

### 1. Latency lÃ  gÃ¬?
LÃ  tá»•ng thá»i gian tá»« lÃºc há»‡ thá»‘ng nháº­n cÃ¢u há»i Ä‘áº¿n lÃºc tráº£ vá» cÃ¢u tráº£ lá»i cuá»‘i cÃ¹ng.

### 2. Äá» xuáº¥t giáº£m latency
Giáº£m `max_tokens`, dÃ¹ng model nhá» hÆ¡n hoáº·c model free nhanh hÆ¡n, cáº¯t bá»›t prompt dÃ i, vÃ  chá»‰ gá»i cÃ¡c agent tháº­t sá»± cáº§n thiáº¿t.

### 3. Demo giáº£m latency nhÆ° tháº¿ nÃ o?
Cháº¡y láº¡i `test_client.py` trÆ°á»›c vÃ  sau khi Ä‘á»•i model/prompt/cap token, rá»“i so sÃ¡nh thá»i gian thá»±c thi Ä‘á»ƒ chá»©ng minh má»©c giáº£m.

## Ghi chÃº

- CÃ¡c cÃ¢u tráº£ lá»i trÃªn lÃ  ngáº¯n gá»n Ä‘á»ƒ dÃ¹ng lÃ m tÃ i liá»‡u Ã´n táº­p.
- Náº¿u cáº§n, cÃ³ thá»ƒ má»Ÿ rá»™ng thÃ nh báº£n chi tiáº¿t hÆ¡n cho tá»«ng bÃ i.
