"""The default TemplateStrategy.generate_result_async must not run its sync work on the loop."""

import asyncio
import time

from verbatim_core.templates.base import TemplateStrategy

ENCODE_SECONDS = 0.5


class SlowTemplateStrategy(TemplateStrategy):
    """A stub whose template generation costs a slow synchronous encode."""

    def generate(self, question, spans, citation_count=0):
        time.sleep(ENCODE_SECONDS)
        return "template"

    def fill(self, template, display_spans, citation_spans):
        return template

    def save_state(self):
        return {}

    def load_state(self, state):
        pass


async def test_other_coroutines_keep_running_during_generate_result_async():
    strategy = SlowTemplateStrategy()

    slow = asyncio.create_task(strategy.generate_result_async("q", ["s"]))
    started = time.perf_counter()
    await asyncio.sleep(0.05)
    elapsed = time.perf_counter() - started

    assert elapsed < ENCODE_SECONDS  # the loop did not stall for the encode
    assert not slow.done()
    await asyncio.wait_for(slow, timeout=5)
