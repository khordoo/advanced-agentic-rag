from fastapi import APIRouter, status

health_router = APIRouter(tags=["System"])


@health_router.get("/", status_code=status.HTTP_200_OK)
def ping():
    """
    Dedicated health check endpoint for AWS Target Groups and Docker lifecycle.
    """
    return {"status": "ok"}
