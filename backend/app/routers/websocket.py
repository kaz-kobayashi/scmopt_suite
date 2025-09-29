"""
WebSocketリアルタイム通信エンドポイント
"""
import json
import asyncio
from typing import Dict, Set
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uuid import uuid4
import logging

from ..models.realtime_models import (
    ScheduleEvent,
    ScheduleEventType,
    RealtimeStats,
    ScheduleAlert
)
from ..services.realtime_schedule_service import realtime_schedule_service

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """WebSocket接続管理クラス"""
    
    def __init__(self):
        # schedule_id -> Set[WebSocket]のマッピング
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # WebSocket -> client_idのマッピング
        self.client_ids: Dict[WebSocket, str] = {}
        # WebSocket -> schedule_idsのマッピング
        self.client_schedules: Dict[WebSocket, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """新しいWebSocket接続を受け入れる"""
        await websocket.accept()
        self.client_ids[websocket] = client_id
        self.client_schedules[websocket] = set()
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, websocket: WebSocket):
        """WebSocket接続を切断"""
        client_id = self.client_ids.get(websocket, "unknown")
        
        # 購読しているスケジュールから削除
        if websocket in self.client_schedules:
            for schedule_id in self.client_schedules[websocket]:
                if schedule_id in self.active_connections:
                    self.active_connections[schedule_id].discard(websocket)
                    if not self.active_connections[schedule_id]:
                        del self.active_connections[schedule_id]
            del self.client_schedules[websocket]
        
        # クライアントIDを削除
        if websocket in self.client_ids:
            del self.client_ids[websocket]
        
        logger.info(f"Client {client_id} disconnected")
    
    async def subscribe_to_schedule(self, websocket: WebSocket, schedule_id: str):
        """特定のスケジュールへの購読"""
        if schedule_id not in self.active_connections:
            self.active_connections[schedule_id] = set()
        
        self.active_connections[schedule_id].add(websocket)
        self.client_schedules[websocket].add(schedule_id)
        
        client_id = self.client_ids.get(websocket, "unknown")
        logger.info(f"Client {client_id} subscribed to schedule {schedule_id}")
    
    async def unsubscribe_from_schedule(self, websocket: WebSocket, schedule_id: str):
        """スケジュールの購読解除"""
        if schedule_id in self.active_connections:
            self.active_connections[schedule_id].discard(websocket)
            if not self.active_connections[schedule_id]:
                del self.active_connections[schedule_id]
        
        if websocket in self.client_schedules:
            self.client_schedules[websocket].discard(schedule_id)
        
        client_id = self.client_ids.get(websocket, "unknown")
        logger.info(f"Client {client_id} unsubscribed from schedule {schedule_id}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """特定のクライアントにメッセージを送信"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
    
    async def broadcast_to_schedule(self, schedule_id: str, message: str):
        """特定のスケジュールを購読している全クライアントにブロードキャスト"""
        if schedule_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[schedule_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to connection: {e}")
                    disconnected.append(connection)
            
            # 切断されたコネクションを削除
            for conn in disconnected:
                self.disconnect(conn)


manager = ConnectionManager()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocketエンドポイント"""
    await manager.connect(websocket, client_id)
    
    try:
        # 初期接続メッセージ
        await manager.send_personal_message(
            json.dumps({
                "type": "connection",
                "status": "connected",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            }),
            websocket
        )
        
        while True:
            # クライアントからのメッセージを待機
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # メッセージタイプに応じた処理
            if message["type"] == "subscribe":
                schedule_id = message["schedule_id"]
                await manager.subscribe_to_schedule(websocket, schedule_id)
                
                # 購読確認メッセージ
                await manager.send_personal_message(
                    json.dumps({
                        "type": "subscription",
                        "status": "subscribed",
                        "schedule_id": schedule_id,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
                
                # 現在の状態を送信
                await send_current_state(websocket, schedule_id)
                
            elif message["type"] == "unsubscribe":
                schedule_id = message["schedule_id"]
                await manager.unsubscribe_from_schedule(websocket, schedule_id)
                
                # 購読解除確認メッセージ
                await manager.send_personal_message(
                    json.dumps({
                        "type": "subscription",
                        "status": "unsubscribed",
                        "schedule_id": schedule_id,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
                
            elif message["type"] == "ping":
                # ピンポン応答
                await manager.send_personal_message(
                    json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
                
            elif message["type"] == "get_stats":
                schedule_id = message["schedule_id"]
                stats = realtime_schedule_service.get_realtime_stats(schedule_id)
                if stats:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "stats_update",
                            "schedule_id": schedule_id,
                            "stats": stats.dict(),
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def send_current_state(websocket: WebSocket, schedule_id: str):
    """現在のスケジュール状態を送信"""
    try:
        # スケジュールの状態を取得
        schedule = realtime_schedule_service.get_schedule_status(schedule_id)
        if schedule:
            await manager.send_personal_message(
                json.dumps({
                    "type": "schedule_state",
                    "schedule_id": schedule_id,
                    "state": {
                        "status": schedule.status,
                        "last_update": schedule.last_update.isoformat(),
                        "active_events_count": len(schedule.active_events),
                        "auto_reoptimize": schedule.auto_reoptimize
                    },
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )
        
        # 統計情報を取得
        stats = realtime_schedule_service.get_realtime_stats(schedule_id)
        if stats:
            await manager.send_personal_message(
                json.dumps({
                    "type": "stats_update",
                    "schedule_id": schedule_id,
                    "stats": stats.dict(),
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )
    except Exception as e:
        logger.error(f"Error sending current state: {e}")


async def notify_event_added(schedule_id: str, event: ScheduleEvent):
    """新しいイベントの通知"""
    message = json.dumps({
        "type": "event_added",
        "schedule_id": schedule_id,
        "event": {
            "id": event.id,
            "event_type": event.event_type,
            "target_id": event.target_id,
            "description": event.description,
            "impact_level": event.impact_level,
            "timestamp": event.timestamp.isoformat()
        },
        "timestamp": datetime.now().isoformat()
    })
    
    await manager.broadcast_to_schedule(schedule_id, message)


async def notify_stats_update(schedule_id: str, stats: RealtimeStats):
    """統計情報更新の通知"""
    message = json.dumps({
        "type": "stats_update",
        "schedule_id": schedule_id,
        "stats": stats.dict(),
        "timestamp": datetime.now().isoformat()
    })
    
    await manager.broadcast_to_schedule(schedule_id, message)


async def notify_alert(schedule_id: str, alert: ScheduleAlert):
    """アラートの通知"""
    message = json.dumps({
        "type": "alert",
        "schedule_id": schedule_id,
        "alert": {
            "id": alert.id,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat()
        },
        "timestamp": datetime.now().isoformat()
    })
    
    await manager.broadcast_to_schedule(schedule_id, message)


async def notify_reoptimization_started(schedule_id: str, request_id: str):
    """再最適化開始の通知"""
    message = json.dumps({
        "type": "reoptimization_started",
        "schedule_id": schedule_id,
        "request_id": request_id,
        "timestamp": datetime.now().isoformat()
    })
    
    await manager.broadcast_to_schedule(schedule_id, message)


async def notify_reoptimization_completed(schedule_id: str, request_id: str, success: bool):
    """再最適化完了の通知"""
    message = json.dumps({
        "type": "reoptimization_completed",
        "schedule_id": schedule_id,
        "request_id": request_id,
        "success": success,
        "timestamp": datetime.now().isoformat()
    })
    
    await manager.broadcast_to_schedule(schedule_id, message)


# バックグラウンドタスク: スケジュール監視
async def monitor_schedules():
    """アクティブなスケジュールを監視し、更新を通知"""
    while True:
        try:
            # アクティブなスケジュールを監視
            for schedule_id in list(manager.active_connections.keys()):
                # 統計情報を取得して送信
                stats = realtime_schedule_service.get_realtime_stats(schedule_id)
                if stats:
                    await notify_stats_update(schedule_id, stats)
            
            # 5秒ごとに更新
            await asyncio.sleep(5)
        
        except Exception as e:
            logger.error(f"Error in schedule monitoring: {e}")
            await asyncio.sleep(5)


# アプリケーション起動時にバックグラウンドタスクを開始
@router.on_event("startup")
async def startup_event():
    """起動時の処理"""
    asyncio.create_task(monitor_schedules())
    logger.info("WebSocket monitoring task started")