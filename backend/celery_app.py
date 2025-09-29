"""
Celery configuration for async VRP processing
"""
from celery import Celery
import os
from kombu import Queue

# Configure Redis connection
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Create Celery app
celery_app = Celery(
    'scmopt_vrp',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['app.tasks.vrp_tasks']
)

# Configure Celery settings
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task routing
    task_routes={
        'app.tasks.vrp_tasks.solve_vrp_async': {'queue': 'vrp_queue'},
        'app.tasks.vrp_tasks.solve_batch_vrp_async': {'queue': 'batch_queue'},
    },
    
    # Queue configuration
    task_default_queue='default',
    task_queues=(
        Queue('default'),
        Queue('vrp_queue'),
        Queue('batch_queue'),
    ),
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    
    # Result settings
    result_expires=3600,  # 1 hour
    
    # Task time limits
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,  # 10 minutes
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Task annotations for specific behavior
celery_app.conf.task_annotations = {
    'app.tasks.vrp_tasks.solve_vrp_async': {
        'rate_limit': '10/m',  # 10 tasks per minute
        'time_limit': 600,     # 10 minutes max
    },
    'app.tasks.vrp_tasks.solve_batch_vrp_async': {
        'rate_limit': '5/m',   # 5 batch tasks per minute
        'time_limit': 1800,    # 30 minutes max
    },
}