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
# TODO 多进程压测