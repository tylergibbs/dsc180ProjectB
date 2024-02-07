


class TimeseriesDataset:

      def __init__(self, n, v, w, WA, seed = 1):
          super().__init__(n, v, v, w+1, w, WA, constraint([]), seed)

