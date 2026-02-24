import sys
import os
import time
import argparse
import subprocess
from collections import deque
from statistics import mean

from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
from pyqtgraph import mkBrush
import zmq
import numpy as np

import logging
import utils.logging_setup as logging_setup


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
    def __init__(self, subscriber_only=False, logger=None):
        super().__init__()
        self.setWindowTitle("RLlib + Engine Monitor")
        if logger is None:
            self.log = logging.getLogger('MyRLApp.GUI')
        else:
            self.log = logger
        self.log.info(f"GUI, PID={os.getpid()}")

        if subscriber_only:
            self.setup_run_proc = None
        else:
            # Standalone mode: launch setup_run.py (which spawns env_runner, minion, etc.)
            algo = 'IMPALA'
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            setup_run_path = os.path.join(script_dir, "setup_run.py")
            cmd = [
                sys.executable,
                setup_run_path,
                "--algo", algo,
                "--enable-zmq", "True",
            ]
            self.setup_run_proc = subprocess.Popen(cmd)

        # create containers for plot parameters
        self.plot_colors = ["#e60049", "#0bb4ff", "#50e991", "#ffa300", "#9b19f5", "#dc0ab4", "#b3d4ff", "#00bfa0"]
        self.plot_line_width = 5

        # data structures for plotting
        self._max_points = 3000
        # engine: dynamic curves keyed by metric name
        self.engine_curves = {}
        self.engine_data = {}
        self.engine_x = deque(maxlen=self._max_points)
        self.engine_count = 0
        self.evaluation_count = 0
        self.evaluation_x = deque(maxlen=self._max_points)

        # manually set fields to be plotted
        self.engine_data["imep"] = deque(maxlen=self._max_points)
        self.engine_data["mprr"] = deque(maxlen=self._max_points)
        self.engine_data["target imep"] = deque(maxlen=self._max_points)
        self.engine_data["mean sampled imep"] = deque(maxlen=self._max_points)
        self.engine_data["evaluation error"] = deque(maxlen=self._max_points)
        self.engine_data["eval imep"] = deque(maxlen=self._max_points)
        self.engine_data["eval target imep"] = deque(maxlen=self._max_points)
        # self.engine_data["ratio_p99"] = deque(maxlen=self._max_points)

        # training/eval reward: curves vs iteration
        self.training_curve = None
        self.eval_reward_curve = None
        self.training_x = []
        self.training_y = []
        self.eval_reward_y = []

        # set up the UI
        central = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        alpha = 225

        # engine metrics (load tracking) plot
        self.load_plot = pg.PlotWidget(title="Load Tracking (minion.py)")
        legend_load = self.load_plot.addLegend()
        legend_load.setBrush(mkBrush(255, 255, 255, alpha))  # RGBA, 200 alpha
        # legend_load.setFrame(True)
        self.load_plot.showGrid(x=True, y=True)
        self.load_plot.setBackground('w')
        vlay.addWidget(self.load_plot)
        # make curves for load tracking plot
        pen = pg.mkPen(color=self.plot_colors[0], width=self.plot_line_width)
        curve = self.load_plot.plot(name='imep', pen=pen)
        self.engine_curves['imep'] = curve
        pen = pg.mkPen(color=self.plot_colors[1], width=self.plot_line_width)
        curve = self.load_plot.plot(name='target imep', pen=pen)
        self.engine_curves['target imep'] = curve
        pen = pg.mkPen(color=self.plot_colors[2], width=self.plot_line_width)
        curve = self.load_plot.plot(name='mean sampled imep', pen=pen)
        self.engine_curves['mean sampled imep'] = curve

        # engine metrics (safety) plot
        self.safety_plot = pg.PlotWidget(title="Safety (minion.py)")
        legend_safety = self.safety_plot.addLegend()
        legend_safety.setBrush(mkBrush(255, 255, 255, alpha))  # RGBA, 200 alpha
        self.safety_plot.showGrid(x=True, y=True)
        self.safety_plot.setBackground('w')
        vlay.addWidget(self.safety_plot)
        # make curve for safety plot
        pen = pg.mkPen(color=self.plot_colors[3], width=self.plot_line_width)
        curve = self.safety_plot.plot(name='mprr', pen=pen)
        self.engine_curves['mprr'] = curve

        # engine metrics (safety) plot
        self.evaluation_plot = pg.PlotWidget(title="Evaluation (minion.py)")
        legend_evaluation = self.evaluation_plot.addLegend()
        legend_evaluation.setBrush(mkBrush(255, 255, 255, alpha))  # RGBA, 200 alpha
        self.evaluation_plot.showGrid(x=True, y=True)
        self.evaluation_plot.setBackground('w')
        vlay.addWidget(self.evaluation_plot)
        # make curve for safety plot
        pen = pg.mkPen(color=self.plot_colors[4], width=self.plot_line_width)
        curve = self.evaluation_plot.plot(name='evaluation error', pen=pen)
        self.engine_curves['evaluation error'] = curve

        # training & evaluation reward plot
        self.training_plot = pg.PlotWidget(title="Reward (custom_run.py)")
        legend_training = self.training_plot.addLegend()
        legend_training.setBrush(mkBrush(255, 255, 255, alpha))  # RGBA, 200 alpha
        self.training_plot.showGrid(x=True, y=True)
        self.training_plot.setBackground('w')
        vlay.addWidget(self.training_plot)
        # *** need to make curves?

        # eval IMEP plot (desired vs actual, eval rollouts)
        self.eval_imep_plot = pg.PlotWidget(title="Eval IMEP Tracking (minion.py)")
        legend_eval_imep = self.eval_imep_plot.addLegend()
        legend_eval_imep.setBrush(mkBrush(255, 255, 255, alpha))  # RGBA, 200 alpha
        self.eval_imep_plot.showGrid(x=True, y=True)
        self.eval_imep_plot.setBackground('w')
        vlay.addWidget(self.eval_imep_plot)
        pen = pg.mkPen(color=self.plot_colors[0], width=self.plot_line_width)
        curve = self.eval_imep_plot.plot(name='eval imep', pen=pen)
        self.engine_curves['eval imep'] = curve
        pen = pg.mkPen(color=self.plot_colors[1], width=self.plot_line_width)
        curve = self.eval_imep_plot.plot(name='eval target imep', pen=pen)
        self.engine_curves['eval target imep'] = curve

        # ── Insert a horizontal layout at the top for controls ──
        controls = QtWidgets.QHBoxLayout()
        self.stop_btn = QtWidgets.QPushButton("Stop Processes")
        self.stop_btn.clicked.connect(self.stop_processes)
        controls.addWidget(self.stop_btn)
        # you could add more buttons here later (e.g. “Pause”, “Restart”)

        # Insert controls above the plots
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.addLayout(controls)
        main_layout.addWidget(self.load_plot)
        main_layout.addWidget(self.safety_plot)
        main_layout.addWidget(self.evaluation_plot)
        main_layout.addWidget(self.training_plot)
        main_layout.addWidget(self.eval_imep_plot)
        self.setCentralWidget(central)

        # 3) Start ZMQ listener thread
        #    (adjust these addresses if you used tcp:// or a different ipc path)
        addresses = [
            "ipc:///tmp/engine.ipc",
            "ipc:///tmp/training.ipc",
        ]
        self.listener = ZmqListener(addresses)
        self.listener.message.connect(self.on_zmq_message)
        self.listener.start()

        # 4a) Setup a QTimer to refresh the high-speed plots at 10 Hz
        self._refresh_timer_HS = QtCore.QTimer(self)
        self._refresh_timer_HS.setInterval(100)  # 100 ms => 10 Hz
        self._refresh_timer_HS.timeout.connect(self._refresh_plots_hs)
        self._refresh_timer_HS.start()

        # 4a) Setup a QTimer to refresh the low-speed plots at 1 Hz
        self._refresh_timer_LS = QtCore.QTimer(self)
        self._refresh_timer_LS.setInterval(1000)  # 1000 ms => 1 Hz
        self._refresh_timer_LS.timeout.connect(self._refresh_plots_ls)
        self._refresh_timer_LS.start()

        self.log.debug("GUI: Done with init.")

    def stop_processes(self):
        if self.listener.isRunning():
            self.listener.stop()

        if self.setup_run_proc is not None and self.setup_run_proc.poll() is None:
            self.setup_run_proc.terminate()
            try:
                self.setup_run_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.setup_run_proc.kill()

    def closeEvent(self, event):
        # Clean up ZMQ thread
        self.listener.stop()
        # Terminate setup_run.py (and thus its children) if we spawned it
        if self.setup_run_proc is not None and self.setup_run_proc.poll() is None:
            self.setup_run_proc.terminate()
        super().closeEvent(event)

    @QtCore.pyqtSlot()
    def _refresh_plots_hs(self):
        x = list(self.engine_x)
        self.engine_curves['imep'].setData(x, list(self.engine_data['imep']))
        self.engine_curves['target imep'].setData(x, list(self.engine_data['target imep']))
        self.engine_curves['mprr'].setData(x, list(self.engine_data['mprr']))
        self.engine_curves['mean sampled imep'].setData(x, list(self.engine_data['mean sampled imep']))

    @QtCore.pyqtSlot()
    def _refresh_plots_ls(self):
        self.engine_curves['evaluation error'].setData(
            list(self.evaluation_x),
            list(self.engine_data['evaluation error'])
        )
        self.engine_curves['eval imep'].setData(
            list(self.evaluation_x),
            list(self.engine_data['eval imep'])
        )
        self.engine_curves['eval target imep'].setData(
            list(self.evaluation_x),
            list(self.engine_data['eval target imep'])
        )
        if self.training_curve is not None:
            self.training_curve.setData(self.training_x, self.training_y)
        if self.eval_reward_curve is not None:
            self.eval_reward_curve.setData(self.training_x, self.eval_reward_y)

    @QtCore.pyqtSlot(dict)
    def on_zmq_message(self, msg):
        self.log.debug(f"GUI: In on_zmq_message.")
        topic = msg.get("topic", "")
        self.log.debug(f"GUI: topic -> {topic}.")
        if topic == "engine":
            self._update_engine(msg)
        elif topic == "training":
            self._update_training(msg)
        elif topic == "evaluation":
            self._update_evaluation(msg)

    def _update_evaluation(self, msg):
        self.log.debug(f"GUI: In _update_evaluation.")
        self.evaluation_count += 1
        self.evaluation_x.append(self.evaluation_count)

        current_imep = msg["current imep"]
        target_imep = msg["target"]

        self.engine_data['eval imep'].append(current_imep)
        self.engine_data['eval target imep'].append(target_imep)
        self.engine_data['evaluation error'].append(np.abs(current_imep - target_imep))
        self.log.debug(f"GUI: Done with _update_evaluation.")

    def _update_engine(self, msg):
        self.log.debug(f"GUI: In _update_engine.")
        self.engine_count += 1
        self.engine_x.append(self.engine_count)
        self.log.debug(f"GUI (_update_engine): msg -> {msg}.")

        data = {
            "imep": msg["current imep"],
            "mprr": msg["mprr"],
            "target imep": msg["target"],
            "mean sampled imep": mean(self.engine_data['target imep']) if list(
                self.engine_data['target imep']) != [] else 0,
        }

        # if list(self.engine_data['target imep']) is not []:
        #     data.update({"mean sampled imep": mean(self.engine_data['target imep'])})

        # self.log.debug(f"GUI (_update_engine): data -> {data}.")
        for i, (k, v) in enumerate(data.items()):
            # append data
            data_list = self.engine_data[k]
            data_list.append(v)

        self.log.debug(f"GUI: Done with _update_engine.")

    def _update_training(self, msg):
        if "iteration" not in msg:
            return
        x = msg["iteration"]

        if "mean_return" not in msg:
            return
        y = msg["mean_return"]

        self.training_x.append(x)
        self.training_y.append(y)

        if len(self.training_x) > self._max_points:
            self.training_x.pop(0)
            self.training_y.pop(0)

        if self.training_curve is None:
            pen = pg.mkPen(color=self.plot_colors[-1], width=self.plot_line_width)
            self.training_curve = self.training_plot.plot(name="train mean_return", pen=pen)
            self.training_plot.addLegend()


def main():
    logger = logging.getLogger('MyRLApp.GUI')
    logger.info(f"GUI, PID={os.getpid()}")
    gui_parser = argparse.ArgumentParser(description="RLlib + Engine Monitor GUI")
    gui_parser.add_argument(
        "--subscriber-only",
        action="store_true",
        help="Run in subscriber-only mode (do not spawn setup_run; expect it to be already running).",
    )
    gui_args, _ = gui_parser.parse_known_args()

    logger.debug(f"GUI: Going to launch application.")

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(subscriber_only=gui_args.subscriber_only, logger=logger)
    win.resize(1200, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
