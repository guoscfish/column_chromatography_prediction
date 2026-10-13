import sys
import free_scoped_request_gate as gate
if __name__=='__main__':
    gate.activate(6101)
    sys.exit(gate.runner.worker(6101))
