from huggingface_hub import HfApi, login

login()

api = HfApi()

api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id="AmbhariiLabs/pulseflow-mlops",
    repo_type="space",
    create_pr=True,
    commit_message="Add PulseFlow Gradio demo app"
)
print("app.py PR created")

api.upload_file(
    path_or_fileobj="hf_requirements.txt",
    path_in_repo="requirements.txt",
    repo_id="AmbhariiLabs/pulseflow-mlops",
    repo_type="space",
    create_pr=True,
    commit_message="Add Space requirements"
)
print("requirements.txt PR created")

print("Done — merge PRs at: https://huggingface.co/spaces/AmbhariiLabs/pulseflow-mlops/discussions")