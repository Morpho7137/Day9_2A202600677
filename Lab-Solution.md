# Báo cáo hoàn thành Day 9 - Multi-Agent MCP/A2A

## Thông tin nộp bài

- Repository nộp bài: `Morpho7137/Day9_2A202600677`
- Mã sinh viên: `2A202600677`
- Họ tên: `Nguyễn Anh Kiệt`
- Code cá nhân từ repository `2A202600677_Nguyen-Anh-Kiet` đã được đặt trong thư mục `Lab_Assignment/`.

## Checklist hoàn thành

### Exercise 2: Tools và Knowledge Base

- Đã thêm entry `labor_law` vào `LEGAL_KNOWLEDGE`.
- Đã tạo tool `check_statute_of_limitations(case_type: str)`.
- Đã thêm tool mới vào danh sách `tools`.
- Đã xử lý tool call cho `check_statute_of_limitations`.
- Đã thêm fallback để không in kết quả rỗng nếu model trả về empty content.

File liên quan: `exercises/exercise_2_tools.py`

### Exercise 4: Multi-Agent với Privacy Agent

- Đã thêm `privacy_analysis` vào state.
- Đã implement `privacy_agent`.
- Đã thêm logic routing cho privacy bằng các từ khóa như `data`, `privacy`, `gdpr`, `du lieu`, `breach`.
- Đã thêm `privacy_agent` vào graph.
- Đã nối edge từ `privacy_agent` về `aggregate_results`.
- Đã tổng hợp `privacy_analysis` trong final report.

File liên quan: `exercises/exercise_4_multiagent.py`

### Stage 5: Distributed A2A

- Đã chạy stack gồm Registry, Customer Agent, Law Agent, Tax Agent, Compliance Agent.
- Đã chạy `test_client.py` thành công.
- Đã cấu hình LLM qua OpenRouter.
- Đã giới hạn output token bằng `OPENROUTER_MAX_TOKENS` để giảm latency và tránh lỗi thiếu credit.

File liên quan:

- `common/llm.py`
- `test_client.py`
- `start_all.sh`

## Kết quả kiểm thử

Các kiểm tra đã thực hiện:

```powershell
py -3.11 -m py_compile common\llm.py exercises\exercise_2_tools.py exercises\exercise_4_multiagent.py
$env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\python.exe exercises\exercise_2_tools.py
$env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\python.exe exercises\exercise_4_multiagent.py
$env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\python.exe test_client.py
```

Kết quả:

- Syntax check passed.
- Exercise 2 chạy được và có output không rỗng.
- Exercise 4 chạy được và tạo báo cáo cuối cùng.
- Stage 5 chạy end-to-end qua A2A client.

## Câu hỏi khái niệm

### Phần 1: Direct LLM Calling

#### 1. LLM được khởi tạo như thế nào?

LLM được khởi tạo bằng hàm `get_llm()` trong `common/llm.py`. Hàm này tạo một đối tượng `ChatOpenAI`, dùng OpenRouter làm OpenAI-compatible endpoint.

#### 2. Message gửi đến LLM có cấu trúc gì?

Message là một danh sách gồm nhiều loại message. Trong Stage 1 có `SystemMessage` để định nghĩa vai trò của model và `HumanMessage` để chứa câu hỏi của người dùng.

#### 3. Vì sao cần `SystemMessage` và `HumanMessage`?

`SystemMessage` đặt bối cảnh, vai trò, phong cách trả lời và ràng buộc hành vi. `HumanMessage` là nội dung người dùng hỏi. Tách hai phần này giúp prompt rõ ràng và dễ kiểm soát hơn.

### Phần 2: LLM + RAG & Tools

#### 1. `@tool` decorator được dùng ở đâu?

`@tool` được dùng trước các hàm muốn expose cho LLM, ví dụ hàm tra cứu knowledge base hoặc tính toán penalty. Sau khi gắn `@tool`, LangChain có thể chuyển function đó thành tool callable.

#### 2. `LEGAL_KNOWLEDGE` được cấu trúc như thế nào?

`LEGAL_KNOWLEDGE` là list các dictionary. Mỗi dictionary có `id`, `keywords`, và `text`. `keywords` dùng để match truy vấn, còn `text` là nội dung pháp lý trả về.

#### 3. LLM bind với tools ra sao?

LLM được bind bằng `llm.bind_tools(tools)`. Sau đó LLM có thể trả về `tool_calls`; chương trình thực thi tool tương ứng rồi đưa kết quả vào lại context cho LLM tổng hợp câu trả lời.

### Phần 3: Single Agent với ReAct

#### 1. `create_react_agent()` là gì?

`create_react_agent()` tạo một agent theo pattern ReAct: Reasoning + Acting. Agent tự quyết định cần gọi tool nào, đọc kết quả, rồi tiếp tục cho đến khi đủ thông tin trả lời.

#### 2. So với Stage 2 khác gì?

Stage 2 phải tự viết vòng lặp gọi tool thủ công. Stage 3 để agent tự điều phối tool calls, nên phù hợp với câu hỏi nhiều bước hơn.

#### 3. Vì sao chỉ cần gọi agent một lần?

Vì vòng lặp suy luận, gọi tool và quan sát kết quả đã nằm bên trong graph của agent. Người dùng chỉ cần gửi input ban đầu.

### Phần 4: Multi-Agent In-Process

#### 1. `State` dùng để làm gì?

`State` là dữ liệu dùng chung giữa các node trong LangGraph. Mỗi agent đọc một phần state và ghi kết quả phân tích của mình vào state.

#### 2. `Send()` API dùng để làm gì?

`Send()` dùng để dispatch nhiều nhánh xử lý song song, ví dụ gọi Tax Agent và Compliance Agent cùng lúc rồi gom kết quả ở node aggregate.

#### 3. Vì sao thêm `privacy_agent`?

Vì một số câu hỏi liên quan đến dữ liệu cá nhân, GDPR, data breach hoặc privacy law. Tách `privacy_agent` giúp phân tích chuyên sâu hơn thay vì để một agent tổng quát xử lý tất cả.

#### 4. Conditional routing hoạt động thế nào?

Routing kiểm tra keyword trong câu hỏi. Nếu có từ khóa liên quan đến tax, compliance hoặc privacy thì graph gửi state đến agent tương ứng. Nếu không cần specialist thì đi thẳng đến aggregate.

### Phần 5: Distributed A2A

#### 1. Request flow đi qua những service nào?

Flow chính là:

```text
User -> Customer Agent -> Registry -> Law Agent -> Registry -> Tax/Compliance Agent -> Law Agent -> Customer Agent -> User
```

#### 2. Registry service dùng để làm gì?

Registry cho phép các agent tự đăng ký capability và endpoint. Agent khác có thể discover endpoint theo task thay vì hardcode URL.

#### 3. Làm sao tránh infinite delegation loop?

Dùng `delegation_depth`, `trace_id`, `context_id`, timeout và giới hạn số hop. Trong repo có `MAX_DELEGATION_DEPTH = 3`.

#### 4. Khi Tax Agent bị dừng thì hệ thống xử lý thế nào?

Law Agent sẽ cố discover và call Tax Agent. Nếu fail, code bắt exception và trả về message dạng tax analysis unavailable, thay vì crash toàn bộ hệ thống.

### Câu hỏi ôn tập

#### 1. Khi nào dùng single agent thay vì multi-agent?

Dùng single agent khi task nhỏ, ít domain, không cần parallelism và không cần specialist riêng.

#### 2. Ưu điểm của A2A so với REST/gRPC thông thường?

A2A chuẩn hóa cách agent mô tả capability, message format, task/context và discovery. REST/gRPC chỉ là transport/API style, không tự định nghĩa semantic cho agent collaboration.

#### 3. Vì sao cần Registry thay vì hardcode URLs?

Registry giúp hệ thống linh hoạt hơn khi agent đổi port, scale nhiều instance hoặc bị thay thế. Hardcode URL đơn giản nhưng khó maintain và khó mở rộng.

#### 4. Cách giảm latency?

Các cách đã áp dụng hoặc đề xuất:

- Giảm `OPENROUTER_MAX_TOKENS`.
- Dùng model nhẹ hơn hoặc free model nhanh hơn.
- Giảm prompt dài.
- Chỉ gọi specialist agent khi routing cho thấy thật sự cần.
- Chạy Tax và Compliance song song bằng `Send()`.

## Bonus latency

Trong lần chạy thực tế, Stage 5 có thể mất vài phút do dùng free model. Cấu hình đã dùng để giảm chi phí và latency:

```env
OPENROUTER_MODEL=openai/gpt-oss-20b:free
OPENROUTER_MAX_TOKENS=128
```

Phương án giảm latency tốt hơn nếu có paid credit:

- Dùng model nhỏ, ổn định, hỗ trợ tool calling tốt.
- Giữ `max_tokens` khoảng `128-512`.
- Tối ưu routing để tránh gọi agent không cần thiết.
- Cache kết quả discover từ Registry.

## Ghi chú nộp bài

- Không commit `.env`.
- `.env.example` chỉ chứa placeholder.
- Code cá nhân từ repo `2A202600677_Nguyen-Anh-Kiet` nằm trong `Lab_Assignment/`.
- Bài Day 9 chính nằm ở root repository.
