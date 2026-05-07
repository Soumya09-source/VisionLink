import queue
import logging
from dataclasses import dataclass, field
from typing import Optional

@dataclass(order=True)
class SpeechItem:
    priority: int
    text: str = field(compare=False)
    direction: str = field(compare=False, default="center")
    interrupt: bool = field(compare=False, default=False)
    # Add a timestamp to break ties in priority (FIFO)
    timestamp: float = field(default=0.0)

class SpeechQueueManager:
    """
    Manages a priority queue of speech items.
    Lower priority number = spoken first.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("SpeechQueueManager")
        self._queue = queue.PriorityQueue()
        self._paused = False

    def put(self, item: SpeechItem) -> None:
        """Adds an item to the priority queue."""
        self._queue.put(item)
        self.logger.debug(f"[QueueManager] Enqueued: '{item.text}' (Priority: {item.priority})")

    def get(self, timeout: Optional[float] = None) -> SpeechItem:
        """Retrieves the highest priority item. Blocks if queue is empty."""
        return self._queue.get(block=True, timeout=timeout)

    def task_done(self) -> None:
        """Marks a previously retrieved item as complete."""
        self._queue.task_done()

    def clear(self) -> None:
        """Flushes all pending items from the queue."""
        cleared_count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                cleared_count += 1
            except queue.Empty:
                break
        if cleared_count > 0:
            self.logger.debug(f"[QueueManager] Cleared {cleared_count} items from queue.")

    def pause(self) -> None:
        """Pauses the queue processing (handled by worker thread checking this flag)."""
        self._paused = True
        self.logger.info("[QueueManager] Queue paused.")

    def resume(self) -> None:
        """Resumes the queue processing."""
        self._paused = False
        self.logger.info("[QueueManager] Queue resumed.")

    @property
    def is_paused(self) -> bool:
        return self._paused
