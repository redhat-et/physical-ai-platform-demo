import asyncio
import logging
import traceback

import websockets.asyncio.server as websocket_server

from openpi_client import msgpack_numpy


logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    def __init__(
        self,
        policy,
        host="0.0.0.0",
        port=8000,
        metadata=None,
    ):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    async def _handler(self, websocket):
        packer = msgpack_numpy.Packer()

        # First message sent to client = metadata
        await websocket.send(
            packer.pack(self._metadata)
        )

        async for message in websocket:
            try:
                if not isinstance(message, bytes):
                    raise RuntimeError(
                        "Expected binary websocket message"
                    )

                obs = msgpack_numpy.unpackb(message)

                result = self._policy.infer(obs)

                await websocket.send(
                    packer.pack(result)
                )

            except Exception:
                logger.exception("Inference failed")

                # Keep compatibility with OpenPI client behavior:
                # text message means server-side error.
                await websocket.send(
                    traceback.format_exc()
                )

    async def _serve(self):
        async with websocket_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ):
            logger.info(
                "WebSocket policy server listening on %s:%d",
                self._host,
                self._port,
            )

            await asyncio.Future()

    def serve_forever(self):
        asyncio.run(self._serve())
