mode=$1
startIndex=$2
generate_device_id=$3
guard_device_id=$4

python cold_decoding.py  \
	--seed 12 \
	--mode $mode \
	--pretrained_model Llama-3-8b-instruct \
	--pretrained_guard_model llama-guard-3-8b \
	--device-id $generate_device_id \
	--guard-device-id $guard_device_id \
	--init-temp 1 \
    --length 20 \
	--max-length 20 \
	--num-iters 200 \
	--min-iters 0 \
	--goal-weight 100 \
    --rej-weight 100 \
	--stepsize 0.1 \
	--noise-iters 1 \
	--win-anneal-iters 1000 \
	--start $startIndex \
	--end 100 \
	--lr-nll-portion 1.0 \
    --topk 10 \
    --output-lgt-temp 1 \
	--verbose \
    --straight-through  \
	--large-noise-iters 50,200,500,1500\
	--large_gs_std  1.0,0.5,0.1,0.01  \
	--stepsize-ratio 1  \
    --batch-size 1 \
    --print-every 1000 \
	--fp16 
	
