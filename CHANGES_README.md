# What changed in this build (fix for "Failed on floor 2: Expecting value: line 1 column 1 (char 0)")

1. Removed the hardcoded OpenAI API key from app.py. The app now reads it from
   the OPENAI_API_KEY environment variable and refuses to start without it.
   -> Revoke the old key in your OpenAI dashboard (it was exposed in the
      original zip) and generate a new one.
   -> `cp .env.example .env`, put your new key in .env, then either
      `export OPENAI_API_KEY=...` before running, or use a tool like
      python-dotenv to load .env automatically.

2. Added extract_json() - a 3-tier fallback JSON parser (direct parse ->
   fenced-block-anywhere -> first "{" to last "}") so a stray sentence GPT-4o
   adds before the JSON no longer crashes the request.

3. Added call_gpt4o_json() - wraps every OpenAI call with:
   - response_format={"type": "json_object"} (forces syntactically valid JSON)
   - temperature=0 (deterministic output)
   - automatic retry with a corrective follow-up message if parsing fails
   - retry with backoff on transient API errors
   - logging of the raw GPT text on every failure (see seefloor_debug.log)

4. ai_analyze_multi no longer aborts the whole multi-floor batch when one
   floor fails - it now returns partial results plus a `warnings` list of
   which floors failed and why.

5. Added image validation (PIL .verify()) before sending files to the API.

6. ai_recommend's prompt now asks for {"recommendations": [...]} instead of a
   bare JSON array, since response_format=json_object requires a top-level
   object.

7. requirements.txt now includes openai, networkx, and Pillow, which the app
   actually imports but which were missing before.

See app_py.diff (if you still have it) for the exact line-by-line changes.
