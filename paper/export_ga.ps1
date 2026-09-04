$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open('E:\projects\pycalphad\fe_surrogate\Graphical_Abstract_fe_surrogate.pptx', $true, $false, $false)
$pres.Export('E:\projects\pycalphad\fe_surrogate\paper\figures\ga_hires', 'PNG', 4000, 2250)
$pres.Close()
$ppt.Quit()
Write-Output EXPORTED-HIRES
