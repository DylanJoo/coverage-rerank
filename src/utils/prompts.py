
def get_prompt_tempate(query, documents):
    return {
        "system": system_prompt,
        "task": task_prompt
    }

# system prompts
system_prompt = \
"""You are a helpful assistant. You will be given a task and you need to provide the best possible answer."""

# task-speiciifc prompts 
task_prompt = \
"""You are a helpful assistant. You will be given a task and you need to provide the best possible answer."""
