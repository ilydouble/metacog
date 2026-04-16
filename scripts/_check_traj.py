import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/math_test_metacog_aime25/trajectories/I-1.traj.json'
with open(path) as f:
    traj = json.load(f)

info = traj.get('info', {})
print(f"exit_status : {info.get('exit_status')}")
print(f"submission  : {repr(info.get('submission',''))[:200]}")
print(f"n_steps     : {info.get('n_calls')}")
print(f"model       : {traj.get('info',{}).get('config',{}).get('model',{}).get('model_name')}")
extra_body = traj.get('info',{}).get('config',{}).get('model',{}).get('model_kwargs',{}).get('extra_body')
api_base   = traj.get('info',{}).get('config',{}).get('model',{}).get('model_kwargs',{}).get('api_base')
print(f"api_base    : {api_base}")
print(f"extra_body  : {extra_body}")
print()

for i, msg in enumerate(traj['messages']):
    role = msg.get('role', '')
    content = msg.get('content') or ''
    tool_calls = msg.get('tool_calls') or []
    extra = msg.get('extra', {}) or {}

    if role == 'system':
        print(f"[{i}] SYSTEM ({len(content)} chars):\n{content[:500]}\n")
    elif role == 'user':
        print(f"[{i}] USER ({len(content)} chars):\n{content[:400]}\n")
    elif role == 'assistant':
        tc_summary = []
        for tc in tool_calls:
            fn = tc.get('function', {})
            tc_summary.append(f"{fn.get('name')}({fn.get('arguments','')[:120]})")
        print(f"[{i}] ASSISTANT content={len(content)} tool_calls={len(tool_calls)}")
        if content:
            print(f"     {content[:500]}")
        for s in tc_summary:
            print(f"     CALL: {s}")
        print()
    elif role == 'tool':
        raw = extra.get('raw_output') or content or ''
        rc  = extra.get('returncode', '?')
        print(f"[{i}] TOOL rc={rc}: {raw[:400]}\n")
    elif role == 'exit':
        print(f"[{i}] EXIT: {content[:200]}\n")
