Write-Host "Running baseline..."
python peft_bert.py --mode baseline --max_steps 1
Write-Host "Running lora..."
python peft_bert.py --mode lora --max_steps 1
Write-Host "Running adapter..."
python peft_bert.py --mode adapter --max_steps 1
Write-Host "Running freeze_selective..."
python peft_bert.py --mode freeze_selective --max_steps 1
Write-Host "Running train_attention..."
python peft_bert.py --mode train_attention --max_steps 1
Write-Host "All experiments finished."
