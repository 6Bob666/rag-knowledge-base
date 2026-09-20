param(
    [double]$Threshold = 0.5,
    [int]$Limit = 500,
    [int]$RerankBatchSize = 4,
    [int]$PredictBatchSize = 4,
    [string]$Name = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv311\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "找不到 .venv311 Python：$Python。请先创建虚拟环境并安装依赖。"
}

$Dataset = Join-Path $ProjectRoot "data\t2retrieval\evaluation_dataset.json"
if (-not (Test-Path -LiteralPath $Dataset)) {
    throw "找不到公开数据评测集：$Dataset。请先运行 prepare_t2_retrieval.py。"
}

if ([string]::IsNullOrWhiteSpace($Name)) {
    $safeThreshold = $Threshold.ToString("0.###", [Globalization.CultureInfo]::InvariantCulture)
    $Name = "t2retrieval-$Limit-threshold-$safeThreshold"
}

Push-Location $ProjectRoot
try {
    & $Python experiment_runner.py run `
        --name $Name `
        --dataset data\t2retrieval\evaluation_dataset.json `
        --collection t2_kb_docs `
        --top-k 3 `
        --threshold $Threshold `
        --limit $Limit `
        --rerank-batch-size $RerankBatchSize `
        --predict-batch-size $PredictBatchSize
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
