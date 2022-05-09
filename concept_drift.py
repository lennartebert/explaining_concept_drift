from abc import ABC


class DriftDetector:
    def __main__(self, log, window_size, window_offset, method='PELT'):
        # segment log into windows
        self.logs = _create_log_windows(log, window_size, window_offset)
        
    def _create_log_windows(self, log, window_size, window_offset):
        
        pass
    
    def get_change_points(self):
        pass
    