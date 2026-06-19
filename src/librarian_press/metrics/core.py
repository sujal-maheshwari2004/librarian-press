"""
core.py — a tiny, dependency-free metrics registry that renders the Prometheus
text exposition format (https://prometheus.io/docs/instrumenting/exposition_formats).

No prometheus_client dependency: just Counter + Gauge with labels, thread-safe,
and a render() that produces scrape-able text. A Prometheus / Grafana Alloy
scraper pulls it from the HTTP endpoint in server.py.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()


def _esc_help(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _esc_label(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(value: float) -> str:
    # integers render without a trailing .0; everything else as a plain float
    if value == int(value):
        return str(int(value))
    return repr(float(value))


class Metric:
    def __init__(self, name: str, documentation: str, mtype: str):
        self.name = name
        self.documentation = documentation
        self.type = mtype
        self._samples: dict[tuple, float] = {}

    @staticmethod
    def _key(labels: dict) -> tuple:
        return tuple(sorted(labels.items()))

    def _render_lines(self) -> list[str]:
        if not self._samples:
            return []
        lines = [
            f"# HELP {self.name} {_esc_help(self.documentation)}",
            f"# TYPE {self.name} {self.type}",
        ]
        for key, val in self._samples.items():
            if key:
                labels = ",".join(f'{k}="{_esc_label(v)}"' for k, v in key)
                lines.append(f"{self.name}{{{labels}}} {_fmt(val)}")
            else:
                lines.append(f"{self.name} {_fmt(val)}")
        return lines


class Gauge(Metric):
    def __init__(self, name: str, documentation: str):
        super().__init__(name, documentation, "gauge")

    def set(self, value: float, **labels):
        with _LOCK:
            self._samples[self._key(labels)] = float(value)


class Counter(Metric):
    def __init__(self, name: str, documentation: str):
        super().__init__(name, documentation, "counter")

    def inc(self, amount: float = 1.0, **labels):
        with _LOCK:
            key = self._key(labels)
            self._samples[key] = self._samples.get(key, 0.0) + float(amount)


class Registry:
    def __init__(self):
        self._metrics: list[Metric] = []

    def register(self, metric: Metric) -> Metric:
        self._metrics.append(metric)
        return metric

    def render(self) -> str:
        out: list[str] = []
        with _LOCK:
            for m in self._metrics:
                out.extend(m._render_lines())
        return "\n".join(out) + "\n"


REGISTRY = Registry()
