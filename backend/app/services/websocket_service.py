"""
WebSocket service for real-time VRP job notifications
"""
import json
import logging
import asyncio
from typing import Dict, Set, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

from app.models.async_models import WebSocketMessage

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time job updates"""
    
    def __init__(self):
        # Store active connections by job_id
        self.job_connections: Dict[str, Set[WebSocket]] = {}
        # Store all active connections
        self.all_connections: Set[WebSocket] = set()
        # Store user-specific connections (if authentication is implemented)
        self.user_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str = None, user_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        # Add to all connections
        self.all_connections.add(websocket)
        
        # Add to job-specific connections if job_id provided
        if job_id:
            if job_id not in self.job_connections:
                self.job_connections[job_id] = set()
            self.job_connections[job_id].add(websocket)
        
        # Add to user-specific connections if user_id provided
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(websocket)
        
        logger.info(f"WebSocket connected: job_id={job_id}, user_id={user_id}")
        
        # Send connection confirmation
        welcome_message = WebSocketMessage(
            message_type="connection_established",
            job_id=job_id or "all",
            data={
                "status": "connected",
                "job_id": job_id,
                "user_id": user_id,
                "message": "WebSocket connection established successfully"
            },
            timestamp=datetime.now().timestamp()
        )
        
        await self._send_to_websocket(websocket, welcome_message)
    
    def disconnect(self, websocket: WebSocket, job_id: str = None, user_id: str = None):
        """Remove a WebSocket connection"""
        # Remove from all connections
        self.all_connections.discard(websocket)
        
        # Remove from job-specific connections
        if job_id and job_id in self.job_connections:
            self.job_connections[job_id].discard(websocket)
            if not self.job_connections[job_id]:
                del self.job_connections[job_id]
        
        # Remove from user-specific connections
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected: job_id={job_id}, user_id={user_id}")
    
    async def send_job_update(self, job_id: str, message_type: str, data: Dict[str, Any]):
        """Send update to all clients subscribed to a specific job"""
        if job_id not in self.job_connections:
            logger.debug(f"No active connections for job {job_id}")
            return
        
        message = WebSocketMessage(
            message_type=message_type,
            job_id=job_id,
            data=data,
            timestamp=datetime.now().timestamp()
        )
        
        # Send to job-specific connections
        connections_to_remove = set()
        for websocket in self.job_connections[job_id]:
            try:
                await self._send_to_websocket(websocket, message)
            except Exception as e:
                logger.error(f"Failed to send message to WebSocket: {e}")
                connections_to_remove.add(websocket)
        
        # Remove failed connections
        for websocket in connections_to_remove:
            self.job_connections[job_id].discard(websocket)
            self.all_connections.discard(websocket)
    
    async def send_to_user(self, user_id: str, message_type: str, job_id: str, data: Dict[str, Any]):
        """Send message to all connections for a specific user"""
        if user_id not in self.user_connections:
            logger.debug(f"No active connections for user {user_id}")
            return
        
        message = WebSocketMessage(
            message_type=message_type,
            job_id=job_id,
            data=data,
            timestamp=datetime.now().timestamp()
        )
        
        # Send to user-specific connections
        connections_to_remove = set()
        for websocket in self.user_connections[user_id]:
            try:
                await self._send_to_websocket(websocket, message)
            except Exception as e:
                logger.error(f"Failed to send message to user {user_id}: {e}")
                connections_to_remove.add(websocket)
        
        # Remove failed connections
        for websocket in connections_to_remove:
            self.user_connections[user_id].discard(websocket)
            self.all_connections.discard(websocket)
    
    async def broadcast(self, message_type: str, job_id: str, data: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.all_connections:
            logger.debug("No active connections for broadcast")
            return
        
        message = WebSocketMessage(
            message_type=message_type,
            job_id=job_id,
            data=data,
            timestamp=datetime.now().timestamp()
        )
        
        # Send to all connections
        connections_to_remove = set()
        for websocket in self.all_connections:
            try:
                await self._send_to_websocket(websocket, message)
            except Exception as e:
                logger.error(f"Failed to broadcast message: {e}")
                connections_to_remove.add(websocket)
        
        # Remove failed connections
        for websocket in connections_to_remove:
            self.all_connections.discard(websocket)
            # Also remove from job and user specific connections
            for job_connections in self.job_connections.values():
                job_connections.discard(websocket)
            for user_connections in self.user_connections.values():
                user_connections.discard(websocket)
    
    async def _send_to_websocket(self, websocket: WebSocket, message: WebSocketMessage):
        """Send message to a specific WebSocket connection"""
        try:
            message_json = message.json()
            await websocket.send_text(message_json)
            logger.debug(f"Sent WebSocket message: {message.message_type} for job {message.job_id}")
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            raise
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections"""
        return {
            "total_connections": len(self.all_connections),
            "job_subscriptions": len(self.job_connections),
            "user_connections": len(self.user_connections),
            "jobs_with_subscribers": list(self.job_connections.keys()),
            "connected_users": list(self.user_connections.keys())
        }

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

# Job status notification helpers
async def notify_job_started(job_id: str, job_data: Dict[str, Any]):
    """Notify clients that a job has started processing"""
    await websocket_manager.send_job_update(
        job_id=job_id,
        message_type="job_started",
        data={
            "status": "processing",
            "message": f"Job {job_id} has started processing",
            **job_data
        }
    )

async def notify_job_progress(job_id: str, progress_data: Dict[str, Any]):
    """Notify clients about job progress updates"""
    await websocket_manager.send_job_update(
        job_id=job_id,
        message_type="job_progress",
        data={
            "status": "processing",
            **progress_data
        }
    )

async def notify_job_completed(job_id: str, result_data: Dict[str, Any]):
    """Notify clients that a job has completed successfully"""
    await websocket_manager.send_job_update(
        job_id=job_id,
        message_type="job_completed",
        data={
            "status": "completed",
            "message": f"Job {job_id} completed successfully",
            **result_data
        }
    )

async def notify_job_failed(job_id: str, error_data: Dict[str, Any]):
    """Notify clients that a job has failed"""
    await websocket_manager.send_job_update(
        job_id=job_id,
        message_type="job_failed",
        data={
            "status": "failed",
            "message": f"Job {job_id} failed",
            **error_data
        }
    )

async def notify_batch_progress(batch_id: str, progress_data: Dict[str, Any]):
    """Notify clients about batch job progress"""
    await websocket_manager.send_job_update(
        job_id=batch_id,
        message_type="batch_progress",
        data={
            "status": "processing",
            **progress_data
        }
    )

async def notify_system_status(status_data: Dict[str, Any]):
    """Broadcast system status updates to all clients"""
    await websocket_manager.broadcast(
        message_type="system_status",
        job_id="system",
        data=status_data
    )