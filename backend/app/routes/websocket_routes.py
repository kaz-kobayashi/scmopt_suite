"""
WebSocket routes for real-time VRP job notifications
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional
import logging
import json

from app.services.websocket_service import websocket_manager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws/jobs/{job_id}")
async def websocket_job_updates(websocket: WebSocket, job_id: str, user_id: Optional[str] = Query(None)):
    """
    WebSocket endpoint for real-time job updates
    
    Clients can connect to receive real-time updates for a specific VRP job:
    - Job status changes (pending -> processing -> completed/failed)
    - Progress updates with percentage and stage information
    - Intermediate results and performance metrics
    - Error notifications and troubleshooting information
    
    Usage:
    - Connect: ws://localhost:8000/ws/jobs/{job_id}
    - With user ID: ws://localhost:8000/ws/jobs/{job_id}?user_id=user123
    """
    try:
        await websocket_manager.connect(websocket, job_id=job_id, user_id=user_id)
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client (ping/pong, subscriptions, etc.)
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    await _handle_client_message(websocket, job_id, user_id, message)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from client: {data}")
                    await websocket.send_text(json.dumps({
                        "error": "Invalid JSON format",
                        "timestamp": websocket_manager.websocket_manager.get_connection_stats()
                    }))
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for job {job_id}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket connection for job {job_id}: {e}")
                await websocket.send_text(json.dumps({
                    "error": f"WebSocket error: {str(e)}",
                    "message_type": "error"
                }))
                break
    
    except Exception as e:
        logger.error(f"Failed to establish WebSocket connection for job {job_id}: {e}")
    
    finally:
        websocket_manager.disconnect(websocket, job_id=job_id, user_id=user_id)

@router.websocket("/ws/global")
async def websocket_global_updates(websocket: WebSocket, user_id: Optional[str] = Query(None)):
    """
    WebSocket endpoint for global system updates
    
    Clients can connect to receive system-wide notifications:
    - System status updates
    - Queue length changes
    - Performance metrics
    - Service health notifications
    
    Usage:
    - Connect: ws://localhost:8000/ws/global
    - With user ID: ws://localhost:8000/ws/global?user_id=user123
    """
    try:
        await websocket_manager.connect(websocket, user_id=user_id)
        
        # Send initial system status
        stats = websocket_manager.get_connection_stats()
        await websocket.send_text(json.dumps({
            "message_type": "system_stats",
            "data": stats,
            "timestamp": websocket_manager.get_connection_stats()
        }))
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    await _handle_global_client_message(websocket, user_id, message)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from global client: {data}")
                    await websocket.send_text(json.dumps({
                        "error": "Invalid JSON format"
                    }))
                
            except WebSocketDisconnect:
                logger.info(f"Global WebSocket disconnected for user {user_id}")
                break
            except Exception as e:
                logger.error(f"Error in global WebSocket connection: {e}")
                break
    
    except Exception as e:
        logger.error(f"Failed to establish global WebSocket connection: {e}")
    
    finally:
        websocket_manager.disconnect(websocket, user_id=user_id)

async def _handle_client_message(websocket: WebSocket, job_id: str, user_id: Optional[str], message: dict):
    """Handle incoming messages from job-specific WebSocket clients"""
    message_type = message.get("type")
    
    if message_type == "ping":
        # Respond to ping with pong
        await websocket.send_text(json.dumps({
            "type": "pong",
            "timestamp": websocket_manager.get_connection_stats()
        }))
    
    elif message_type == "subscribe_progress":
        # Client explicitly requesting progress updates
        await websocket.send_text(json.dumps({
            "type": "subscription_confirmed",
            "job_id": job_id,
            "subscription": "progress_updates",
            "message": f"Subscribed to progress updates for job {job_id}"
        }))
    
    elif message_type == "get_status":
        # Client requesting current job status
        # This would typically query the job storage or Celery task
        await websocket.send_text(json.dumps({
            "type": "status_response",
            "job_id": job_id,
            "message": "Status query received - implementation depends on job storage system"
        }))
    
    else:
        logger.warning(f"Unknown message type from client: {message_type}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        }))

async def _handle_global_client_message(websocket: WebSocket, user_id: Optional[str], message: dict):
    """Handle incoming messages from global WebSocket clients"""
    message_type = message.get("type")
    
    if message_type == "ping":
        await websocket.send_text(json.dumps({
            "type": "pong",
            "timestamp": websocket_manager.get_connection_stats()
        }))
    
    elif message_type == "get_stats":
        stats = websocket_manager.get_connection_stats()
        await websocket.send_text(json.dumps({
            "type": "stats_response",
            "data": stats
        }))
    
    elif message_type == "subscribe_system":
        await websocket.send_text(json.dumps({
            "type": "subscription_confirmed",
            "subscription": "system_updates",
            "message": "Subscribed to system-wide updates"
        }))
    
    else:
        logger.warning(f"Unknown global message type: {message_type}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        }))

@router.get("/ws/stats")
async def get_websocket_stats():
    """
    Get current WebSocket connection statistics
    
    Returns information about:
    - Number of active connections
    - Job subscriptions
    - User connections
    - System performance metrics
    """
    try:
        stats = websocket_manager.get_connection_stats()
        return {
            "websocket_stats": stats,
            "status": "healthy",
            "message": "WebSocket service is operational"
        }
    except Exception as e:
        logger.error(f"Failed to get WebSocket stats: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to retrieve WebSocket statistics"
        }