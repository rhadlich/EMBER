import sys
import os
import subprocess

from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
import zmq

import logging
import logging_setup


# ---------- ZMQ subscriber thread ----------
class ZmqListener(QtCore.QThread):
    """
    SUBscribes to both engine and training publishers and emits each message as a dict.
    """
    message = QtCore.pyqtSignal(dict)

    def __init__(self, addresses, parent=None):
        super().__init__(parent)
        self._addresses = addresses
        self._running = True

        self.log = logging.getLogger('MyRLApp.GUI')

    def run(self):
        self.log.debug("ZmqListener: Starting ZMQ Listener thread")
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        # connect to each publisher
        for addr in self._addresses:
            sub.connect(addr)
        # sub.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all topics

        self.log.debug("ZmqListener: Subscribed to addresses.")

        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)

        self.log.debug("ZmqListener: Going into listening loop.")
        while self._running:
            socks = dict(poller.poll(timeout=500))  # 0.5 s timeout so we can shut down cleanly
            if sub in socks and socks[sub] == zmq.POLLIN:
                msg = sub.recv_json()
                if not isinstance(msg, dict):
                    self.log.debug(f"ZmqListener: Received invalid message from engine -> {msg}")
                    continue
                self.message.emit(msg)

        self.log.debug("ZmqListener: Exited listening loop.")
        sub.close()
        ctx.term()

    def stop(self):
        self._running = False
        self.wait()


# ---------- Main application window ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RLlib + Engine Monitor")
        self.log = logging.getLogger('MyRLApp.GUI')
        self.log.info(f"GUI, PID={os.getpid()}")

        # 1) Launch Master.py (which you’ve set up to spawn
        #    custom_run.py, shared_memory_env_runner.py, Minion.py)
        script_dir = os.path.dirname('/Users/rodrigohadlich/PycharmProjects/RayProject/')
        master_path = os.path.join(script_dir, "Master.py")
        # use the same Python interpreter
        self.master_proc = subprocess.Popen([sys.executable, master_path])

        # 2) Set up the UI
        central = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        # 2a) Engine metrics plot
        self.engine_plot = pg.PlotWidget(title="Engine Metrics (Minion.py)")
        self.engine_plot.addLegend()
        self.engine_plot.showGrid(x=True, y=True)
        self.engine_plot.setBackground('w')
        vlay.addWidget(self.engine_plot)

        # 2b) Training reward plot
        self.training_plot = pg.PlotWidget(title="Training Reward (custom_run.py)")
        self.training_plot.showGrid(x=True, y=True)
        self.training_plot.setBackground('w')
        vlay.addWidget(self.training_plot)

        # Create containers for plot parameters
        self.plot_colors = ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']
        self.plot_line_width = 5

        # Data structures for plotting
        self._max_points = 10000
        # engine: dynamic curves keyed by metric name
        self.engine_curves = {}
        self.engine_data = {}
        self.engine_x = []
        self.engine_count = 0

        # manually set fields to be plotted
        self.engine_data["imep"] = []
        self.engine_data["mprr"] = []
        self.engine_data["target imep"] = []

        # training: one curve for reward vs iteration
        self.training_curve = None
        self.training_x = []
        self.training_y = []

        # 3) Start ZMQ listener thread
        #    (adjust these addresses if you used tcp:// or a different ipc path)
        addresses = [
            "ipc:///tmp/engine.ipc",
            "ipc:///tmp/training.ipc",
        ]
        self.listener = ZmqListener(addresses)
        self.listener.message.connect(self.on_zmq_message)
        self.listener.start()

        self.log.debug("GUI: Done with init.")

    def closeEvent(self, event):
        # Clean up ZMQ thread
        self.listener.stop()
        # Terminate Master.py (and thus its children)
        if self.master_proc.poll() is None:
            self.master_proc.terminate()
        super().closeEvent(event)

    @QtCore.pyqtSlot(dict)
    def on_zmq_message(self, msg):
        topic = msg.get("topic", "")
        if topic == "engine":
            self._update_engine(msg)
        elif topic == "training":
            self._update_training(msg)

    def _update_engine(self, msg):
        # self.log.debug(f"GUI: In _update_engine.")
        self.engine_count += 1
        self.engine_x.append(self.engine_count)
        if len(self.engine_x) > self._max_points:
            self.engine_x.pop(0)
        # self.log.debug(f"GUI (_update_engine): msg -> {msg}.")

        data = {
            "imep": msg["state"][0],
            "mprr": msg["state"][1],
            "target imep": msg["target"]
        }

        # self.log.debug(f"GUI (_update_engine): data -> {data}.")
        for i, (k, v) in enumerate(data.items()):
            # create curve on first sighting
            if k not in self.engine_curves:
                pen = pg.mkPen(color=self.plot_colors[i], width=self.plot_line_width)
                curve = self.engine_plot.plot(name=k, pen=pen)
                self.engine_curves[k] = curve

            # append data
            data_list = self.engine_data[k]
            data_list.append(v)
            if len(data_list) > self._max_points:
                data_list.pop(0)
            # redraw
            self.engine_curves[k].setData(self.engine_x, data_list)

    def _update_training(self, msg):
        # Determine iteration & reward keys
        # Adjust these if you used different JSON keys

        self.log.debug(f"GUI (_update_engine): msg -> {msg}.")

        if "iteration" in msg:
            x = msg["iteration"]
        else:
            return

        if "mean_return" in msg:
            y = msg["mean_return"]
        else:
            return

        if "eval_return" in msg:
            y2 = msg["eval_return"]

        self.training_x.append(x)
        self.training_y.append(y)
        if len(self.training_x) > self._max_points:
            self.training_x.pop(0)
            self.training_y.pop(0)

        if self.training_curve is None:
            # first time: create it
            pen = pg.mkPen(color=self.plot_colors[-1], width=self.plot_line_width)
            self.training_curve = self.training_plot.plot(name="Reward", pen=pen)
            self.training_plot.addLegend()

        self.training_curve.setData(self.training_x, self.training_y)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
