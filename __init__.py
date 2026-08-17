"""Native ComfyUI registration for MiniMax H3 one-frame and middle-frame nodes."""

from comfy_api.latest import ComfyExtension

if __package__:
    from .nodes import (
        EmptyMiniMaxH3OneFrameLatent,
        MiniMaxH3ReferenceToImageOneFrame,
        MM1FramePickMiddleFrame,
    )
else:
    from nodes import EmptyMiniMaxH3OneFrameLatent, MiniMaxH3ReferenceToImageOneFrame, MM1FramePickMiddleFrame


class MiniMaxH3OneFrameExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            EmptyMiniMaxH3OneFrameLatent,
            MiniMaxH3ReferenceToImageOneFrame,
            MM1FramePickMiddleFrame,
        ]


async def comfy_entrypoint() -> MiniMaxH3OneFrameExtension:
    return MiniMaxH3OneFrameExtension()


__all__ = ["comfy_entrypoint"]
