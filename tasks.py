from celery import Celery

celery_app = Celery('DocumentProcessor', 
backend='redis://localhost:6379/0',#results will be stored in redis
broker='redis://localhost:6379/0') # 0 redis has 16 databases by default we are using 0th database tasks are queued here
