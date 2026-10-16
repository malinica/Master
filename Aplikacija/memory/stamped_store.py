from langgraph.store.base import BaseStore, PutOp


class KindStampingStore(BaseStore):
    def __init__(self, inner: BaseStore, kind: str):
        self._inner = inner
        self._kind = kind

    def _stamp(self, ops):
        return [
            op._replace(value={**op.value, "kind": self._kind})
            if isinstance(op, PutOp) and op.value is not None
            else op
            for op in ops
        ]

    def batch(self, ops):
        return self._inner.batch(self._stamp(ops))

    async def abatch(self, ops):
        return await self._inner.abatch(self._stamp(ops))
