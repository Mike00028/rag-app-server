from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure, worker_process_init
from src.config.index import appConfig


from src.rag.ingestion.index import process_document

celery_app = Celery(
    "multi-modal-rag",  # Name of the Celery App
    broker=appConfig["redis_url"],  # broker - Redis Queue - Tasks are queued
)

# Disable Celery's logging hijacking to preserve structlog JSON output
celery_app.conf.update(
    worker_hijack_root_logger=False,  # Don't let Celery reconfigure root logger
    worker_log_format='%(message)s',  # Simple format - just the message
    worker_task_log_format='%(message)s',  # Same for task logs
    worker_redirect_stdouts=False,  # Don't redirect stdout/stderr
    worker_redirect_stdouts_level='WARNING',  # If redirected, use WARNING level
)

@worker_process_init.connect
def init_worker_process(sender=None, **kwargs):
    print(f"Celery worker started: {sender}")


@task_prerun.connect
def task_prerun_handler(task_id=None, task=None, args=None, kwargs=None, **extra):
    task_name = task.name if task else "Unknown"
    print(f"Task started: {task_id} - {task_name}")


@task_postrun.connect
def task_postrun_handler(task_id=None, task=None, retval=None, state=None, **_kwargs):
    task_name = task.name if task else "Unknown"
    print(f"Task completed: {task_id} - {task_name} - State: {state}")


@task_failure.connect
def task_failure_handler(task_id=None, exception=None, sender=None, **_kwargs):
    task_name = sender.name if sender else "Unknown"
    print(f"Task failed: {task_id} - {task_name} - Error: {exception}")


@celery_app.task
def perform_rag_ingestion_task(document_id: str, project_id: str, clerk_id: str) -> str :
    print(f"Processing document: {document_id}")
    try:
        process_document_result = process_document(document_id, project_id, clerk_id)
        chunks_created = process_document_result.get("chunks_created")
        print(f"Document processed successfully: {document_id} - Chunks created: {chunks_created}")
        return f"Document {process_document_result['document_id']} processed successfully"
    except Exception as e:
        print(f"Document processing failed: {document_id} - Error: {str(e)}")
        return f"Failed to process document {document_id}: {str(e)}"