# docx → Word 97-2003 .doc (MS Word). 사용: powershell -File save_doc.ps1 <docx 절대경로>
param([string]$Docx)
$doc = [System.IO.Path]::ChangeExtension($Docx, '.doc')
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $d = $word.Documents.Open($Docx, $false, $true)
    $d.SaveAs2([ref]$doc, [ref]0)   # 0 = wdFormatDocument (97-2003)
    $pages = $d.ComputeStatistics(2)
    $d.Close([ref]0)
    Write-Output "DOC: $doc"
    Write-Output "PAGES: $pages"
} finally {
    $word.Quit()
}
