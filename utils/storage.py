import os
import sqlite3
import json
import time
import uuid
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
from datetime import datetime, timedelta

from config.settings import Config


class InferenceStorage:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(Config.DATA_DIR, "inference.db")
        
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classification_results (
                    id TEXT PRIMARY KEY,
                    image_id TEXT,
                    image_path TEXT,
                    top_class_id INTEGER,
                    top_class_name TEXT,
                    top_confidence REAL,
                    all_results TEXT,
                    inference_time REAL,
                    model_version TEXT,
                    created_at REAL,
                    created_at_str TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_results (
                    id TEXT PRIMARY KEY,
                    image_id TEXT,
                    image_path TEXT,
                    num_detections INTEGER,
                    all_detections TEXT,
                    inference_time REAL,
                    model_version TEXT,
                    conf_threshold REAL,
                    iou_threshold REAL,
                    created_at REAL,
                    created_at_str TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_items (
                    id TEXT PRIMARY KEY,
                    detection_result_id TEXT,
                    class_id INTEGER,
                    class_name TEXT,
                    confidence REAL,
                    bbox_x1 REAL,
                    bbox_y1 REAL,
                    bbox_x2 REAL,
                    bbox_y2 REAL,
                    FOREIGN KEY (detection_result_id) REFERENCES detection_results(id)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_classification_created 
                ON classification_results(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_detection_created 
                ON detection_results(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_class_name 
                ON classification_results(top_class_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_detection_class 
                ON detection_items(class_name)
            """)
    
    def save_classification_result(self, image_path: str = None,
                                    image_id: str = None,
                                    results: List[Dict] = None,
                                    inference_time: float = 0.0,
                                    model_version: str = "v1") -> str:
        result_id = str(uuid.uuid4())
        created_at = time.time()
        created_at_str = datetime.fromtimestamp(created_at).isoformat()
        
        top_result = results[0] if results else {}
        top_class_id = top_result.get("class_id", -1)
        top_class_name = top_result.get("class_name", "")
        top_confidence = top_result.get("confidence", 0.0)
        
        all_results_json = json.dumps(results, ensure_ascii=False)
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO classification_results 
                (id, image_id, image_path, top_class_id, top_class_name, 
                 top_confidence, all_results, inference_time, 
                 model_version, created_at, created_at_str)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result_id, image_id, image_path, top_class_id,
                top_class_name, top_confidence, all_results_json,
                inference_time, model_version, created_at, created_at_str,
            ))
        
        return result_id
    
    def save_detection_result(self, image_path: str = None,
                               image_id: str = None,
                               detections: List[Dict] = None,
                               inference_time: float = 0.0,
                               conf_threshold: float = 0.25,
                               iou_threshold: float = 0.45,
                               model_version: str = "v1") -> str:
        result_id = str(uuid.uuid4())
        created_at = time.time()
        created_at_str = datetime.fromtimestamp(created_at).isoformat()
        
        num_detections = len(detections) if detections else 0
        all_detections_json = json.dumps(detections, ensure_ascii=False)
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO detection_results 
                (id, image_id, image_path, num_detections, all_detections, 
                 inference_time, model_version, conf_threshold, 
                 iou_threshold, created_at, created_at_str)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result_id, image_id, image_path, num_detections,
                all_detections_json, inference_time, model_version,
                conf_threshold, iou_threshold, created_at, created_at_str,
            ))
            
            if detections:
                for det in detections:
                    item_id = str(uuid.uuid4())
                    bbox = det.get("bbox", [0, 0, 0, 0])
                    cursor.execute("""
                        INSERT INTO detection_items 
                        (id, detection_result_id, class_id, class_name, 
                         confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        item_id, result_id,
                        det.get("class_id", -1),
                        det.get("class_name", ""),
                        det.get("confidence", 0.0),
                        bbox[0], bbox[1], bbox[2], bbox[3],
                    ))
        
        return result_id
    
    def get_classification_result(self, result_id: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM classification_results WHERE id = ?
            """, (result_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            return self._row_to_classification_dict(row)
    
    def get_detection_result(self, result_id: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM detection_results WHERE id = ?
            """, (result_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            return self._row_to_detection_dict(row)
    
    def list_classification_results(self, class_name: str = None,
                                     min_confidence: float = None,
                                     start_time: float = None,
                                     end_time: float = None,
                                     limit: int = 50,
                                     offset: int = 0) -> List[Dict]:
        query = "SELECT * FROM classification_results WHERE 1=1"
        params = []
        
        if class_name:
            query += " AND top_class_name = ?"
            params.append(class_name)
        
        if min_confidence is not None:
            query += " AND top_confidence >= ?"
            params.append(min_confidence)
        
        if start_time is not None:
            query += " AND created_at >= ?"
            params.append(start_time)
        
        if end_time is not None:
            query += " AND created_at <= ?"
            params.append(end_time)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_classification_dict(row) for row in rows]
    
    def list_detection_results(self, class_name: str = None,
                                min_confidence: float = None,
                                start_time: float = None,
                                end_time: float = None,
                                limit: int = 50,
                                offset: int = 0) -> List[Dict]:
        query = "SELECT DISTINCT dr.* FROM detection_results dr"
        params = []
        where_clauses = []
        
        if class_name:
            query += " JOIN detection_items di ON di.detection_result_id = dr.id"
            where_clauses.append("di.class_name = ?")
            params.append(class_name)
        
        if min_confidence is not None:
            if class_name:
                where_clauses.append("di.confidence >= ?")
            params.append(min_confidence)
        
        if start_time is not None:
            where_clauses.append("dr.created_at >= ?")
            params.append(start_time)
        
        if end_time is not None:
            where_clauses.append("dr.created_at <= ?")
            params.append(end_time)
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        query += " ORDER BY dr.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_detection_dict(row) for row in rows]
    
    def get_classification_stats(self, start_time: float = None,
                                  end_time: float = None) -> Dict:
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = time.time()
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) as total, 
                       AVG(inference_time) as avg_time,
                       AVG(top_confidence) as avg_confidence
                FROM classification_results
                WHERE created_at >= ? AND created_at <= ?
            """, (start_time, end_time))
            row = cursor.fetchone()
            
            cursor.execute("""
                SELECT top_class_name, COUNT(*) as count
                FROM classification_results
                WHERE created_at >= ? AND created_at <= ?
                GROUP BY top_class_name
                ORDER BY count DESC
                LIMIT 20
            """, (start_time, end_time))
            class_rows = cursor.fetchall()
            
            class_distribution = [
                {"class_name": r["top_class_name"], "count": r["count"]}
                for r in class_rows
            ]
            
            return {
                "total_count": row["total"] or 0,
                "avg_inference_time": row["avg_time"] or 0.0,
                "avg_confidence": row["avg_confidence"] or 0.0,
                "class_distribution": class_distribution,
            }
    
    def get_detection_stats(self, start_time: float = None,
                             end_time: float = None) -> Dict:
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = time.time()
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) as total, 
                       AVG(inference_time) as avg_time,
                       AVG(num_detections) as avg_detections
                FROM detection_results
                WHERE created_at >= ? AND created_at <= ?
            """, (start_time, end_time))
            row = cursor.fetchone()
            
            cursor.execute("""
                SELECT di.class_name, COUNT(*) as count
                FROM detection_items di
                JOIN detection_results dr ON dr.id = di.detection_result_id
                WHERE dr.created_at >= ? AND dr.created_at <= ?
                GROUP BY di.class_name
                ORDER BY count DESC
                LIMIT 20
            """, (start_time, end_time))
            class_rows = cursor.fetchall()
            
            class_distribution = [
                {"class_name": r["class_name"], "count": r["count"]}
                for r in class_rows
            ]
            
            return {
                "total_count": row["total"] or 0,
                "avg_inference_time": row["avg_time"] or 0.0,
                "avg_detections_per_image": row["avg_detections"] or 0.0,
                "class_distribution": class_distribution,
            }
    
    def delete_old_results(self, older_than_days: int = 30) -> Dict:
        cutoff = time.time() - older_than_days * 86400
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM detection_items 
                WHERE detection_result_id IN (
                    SELECT id FROM detection_results WHERE created_at < ?
                )
            """, (cutoff,))
            det_items_deleted = cursor.rowcount
            
            cursor.execute("""
                DELETE FROM detection_results WHERE created_at < ?
            """, (cutoff,))
            det_deleted = cursor.rowcount
            
            cursor.execute("""
                DELETE FROM classification_results WHERE created_at < ?
            """, (cutoff,))
            cls_deleted = cursor.rowcount
            
            return {
                "classification_deleted": cls_deleted,
                "detection_deleted": det_deleted,
                "detection_items_deleted": det_items_deleted,
            }
    
    def _row_to_classification_dict(self, row) -> Dict:
        return {
            "id": row["id"],
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "top_class_id": row["top_class_id"],
            "top_class_name": row["top_class_name"],
            "top_confidence": row["top_confidence"],
            "all_results": json.loads(row["all_results"]) if row["all_results"] else [],
            "inference_time": row["inference_time"],
            "model_version": row["model_version"],
            "created_at": row["created_at"],
            "created_at_str": row["created_at_str"],
        }
    
    def _row_to_detection_dict(self, row) -> Dict:
        return {
            "id": row["id"],
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "num_detections": row["num_detections"],
            "all_detections": json.loads(row["all_detections"]) if row["all_detections"] else [],
            "inference_time": row["inference_time"],
            "model_version": row["model_version"],
            "conf_threshold": row["conf_threshold"],
            "iou_threshold": row["iou_threshold"],
            "created_at": row["created_at"],
            "created_at_str": row["created_at_str"],
        }
