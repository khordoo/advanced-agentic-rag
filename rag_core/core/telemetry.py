import os
from langfuse.langchain import CallbackHandler

# TODO: Remove this after testing
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-admin-001"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-admin-001"
os.environ["LANGFUSE_BASE_URL"] = "http://localhost:3000"


def get_observability_callbacks():
    """Returns Langfuse callbacks ONLY if the user has configured them."""
    callbacks = []

    # Check if the user explicitly opted in via their .env file
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            langfuse_handler = CallbackHandler(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                # secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                # host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            )
            callbacks.append(langfuse_handler)
            print("Langfuse observability enabled.")
        except Exception as e:
            print(f"Warning: Langfuse config found but failed to initialize: {e}")

    return callbacks
