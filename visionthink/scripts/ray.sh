# #!/bin/bash


IP=$(hostname -I | awk '{print $1}')
# RAY_ADDRESS="http://${IP}:8888"


ray start --head --port=8888 --node-ip-address=$(hostname -I | awk '{print $1}') --dashboard-host=0.0.0.0

# # HEAD_IP=22.1.186.71  # 这里为head节点的IP，也就是机器A的IP
# HEAD_IP=22.0.200.74
# LOCAL_IP=${HEAD_IP}  # 这里为本机ip
# PORT=8288  # 这里的port需要和前面的保持一致

# # ray status
# # 判断本机IP是否为Head节点的IP
# if [ "$LOCAL_IP" == "$HEAD_IP" ]; then
#     echo "本机 $LOCAL_IP 是Head节点，启动Head节点..."
#     ray start --head --port=$PORT --node-ip-address=$HEAD_IP --dashboard-host=0.0.0.0
# else
#     echo "本机 $LOCAL_IP 是Worker节点，连接到Head节点 $HEAD_IP..."
#     ray start --address=$HEAD_IP:$PORT
# fi


# ray start --head --port=8288 --node-ip-address=22.0.177.63 --dashboard-host=0.0.0.0

# ray start --address=22.0.185.67:8888

# ray start --address=22.0.56.73:8888

# ray start --address=22.0.253.74:8888

# ray start --address=22.1.175.71:8888

# ray start --address=22.0.103.78:8888

# ray start --address=22.0.189.74:8888

# ray start --address=22.0.174.69:8888