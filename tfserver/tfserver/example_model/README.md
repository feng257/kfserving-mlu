# inception_v3 test
```bash
sudo docker exec -it Cambricon-MLU_pcl01_v10.1.0 bash
cd /home/Cambricon-Zy
source ./env.sh

# 启动tfserver
cd /home/Cambricon-Zy/jx/kfserving-mlu/tfserver/tfserver
python ./__main__.py --model_file=/home/Cambricon-Zy/tensorflow/models/online/inception_v3/inception_v3_float16_sparse.mlu.pb --workers=1 --model_name=inception_v3 --input_name=input --output_name=InceptionV3/Predictions/Reshape_1

# 启动inception_v3 http client
cd /home/Cambricon-Zy/jx/kfserving-mlu/tfserver/tfserver/example_model
# 在预测第一张图片是会比较慢，之后预测的时间会变快
python ./inception_v3_http_client.py --model_name=inception_v3 --host=127.0.0.1:8080 --data_dir=/home/Cambricon-Zy/datasets/data_test --num_tests=1 --concurrency=1
python ./inception_v3_http_client.py --model_name=inception_v3 --host=127.0.0.1:8080 --data_dir=/home/Cambricon-Zy/datasets/data_test --num_tests=10 --concurrency=1
```
# 测试结果
```
# 第一次预测（单位：秒）
('Time:', 22.112077)
# 以后预测10次耗时
('Time:', 5.337744)
# 之后预测100次耗时
('Time:', 54.399587)

平均单次请求约540毫秒，约2FPS
# 之后预测100次耗时,8线程
('Time:', 52.098553)
```
# 多进程压测
```
# 使用wrk压测
# 1、生成测试数据，执行generator_wrk_post_lua函数
python inception_v3_http_client.py
# wrk压测 16线程100连接
wrk -t 16 -c 100 -d 30s --latency --timeout 5s -s inception_v3_post.lua http://127.0.0.1:8080/v1/models/inception_v3:predict
```
```
Running 30s test @ http://127.0.0.1:8080/v1/models/inception_v3:predict
  16 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     2.63s     1.46s    4.91s    58.82%
    Req/Sec     2.81      1.42     5.00     69.44%
  Latency Distribution
     50%    2.69s
     75%    3.79s
     90%    4.62s
     99%    4.91s
  108 requests in 30.10s, 2.43MB read
  Socket errors: connect 0, read 0, write 0, timeout 91
Requests/sec:      3.59
Transfer/sec:     82.52KB
```
# TODO 问题：mlu100利用率太低
# TODO 问题：tfserver单worker server端并发性能差