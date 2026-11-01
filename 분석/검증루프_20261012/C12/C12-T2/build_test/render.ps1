# docx → PDF (MS Word) → 쪽 이미지(PyMuPDF). 사용: powershell -File render.ps1 <docx 절대경로>
param([string]$Docx)
$pdf = [System.IO.Path]::ChangeExtension($Docx, '.pdf')
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $d = $word.Documents.Open($Docx, $false, $true)
    $d.SaveAs([ref]$pdf, [ref]17)   # 17 = wdFormatPDF
    $pages = $d.ComputeStatistics(2) # 2 = wdStatisticPages
    $d.Close([ref]0)
    Write-Output "PDF: $pdf"
    Write-Output "PAGES: $pages"
} finally {
    $word.Quit()
}
