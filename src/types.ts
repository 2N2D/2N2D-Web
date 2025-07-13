export interface RobotDataResponse {
  summary: Summary;
  results: Results;
}

export type DataRow = Record<string, number | string>;

export interface Summary {
  rows: number;
  columns: number;
  filename: string;
  missingValues: Record<string, number>;
  dtypes: Record<string, string>;
}

export interface Results {
  columnTypes: ColumnTypes;
  basicInfo: BasicInfo;
  encodingFeasibility: EncodingFeasibility;
  visualizationData: VisualizationData;
}

export interface ColumnTypes {
  categorical: number[];
  numerical: string[];
}

export interface BasicInfo {
  shape: number[];
  missingValues: Record<string, number>;
  dtypes: Record<string, string>;
}

export interface EncodingFeasibility {
  isSafeForOnehot: boolean;
  overallRecommendation: string;
  warnings: number[];
  recommendations: number[];
  highCardinalityColumns: number[];
  memoryEstimate: MemoryEstimate;
  columnEstimate: ColumnEstimate;
  categoricalSummary: CategoricalSummary;
}

export interface MemoryEstimate {
  currentMemoryMb: number;
  estimatedMemoryMb: number;
  memoryIncreaseFactor: number;
}

export interface ColumnEstimate {
  currentColumns: number;
  estimatedFinalColumns: number;
  newColumnsAdded: number;
}

export interface CategoricalSummary {
  totalCategorical: number;
  safeForOnehot: number;
  riskyForOnehot: number;
}

export interface VisualizationData {
  correlationMatrix: CorrelationMatrix;
  distributionData: Record<string, Distribution>;
  categoricalDistributions: Record<string, unknown>; // Empty object in the example
  missingDataHeatmap: MissingDataHeatmap;
  basicStats: BasicStats;
}

export interface CorrelationMatrix {
  data: number[][];
  columns: string[];
  index: string[];
}

export interface Distribution {
  values: number[];
  bins: number[];
  counts: number[];
  quartiles: number[];
  mean: number;
  std: number;
}

export interface MissingDataHeatmap {
  data: number[][];
  columns: string[];
  missingCounts: number[];
  missingPercentages: number[];
}

export interface BasicStats {
  totalRows: number;
  totalColumns: number;
  categoricalColumns: number;
  numericalColumns: number;
  missingValuesTotal: number;
  duplicateRows: number;
}
