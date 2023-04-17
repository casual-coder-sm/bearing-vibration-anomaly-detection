import awsiot.greengrasscoreipc
import awsiot.greengrasscoreipc.client as client
from awsiot.greengrasscoreipc.model import (
    QOS,
    PublishToIoTCoreRequest
)
import time

TIMEOUT = 10

print('....Begin....')
ipc_client = awsiot.greengrasscoreipc.connect()
print('...After connect...')

topic = "collect_device_data/device_reading"
message = "Reading Value = "
clientType = "Mimic Bearing Anomaly Sensor"
qos = QOS.AT_LEAST_ONCE

counter = 1
while True:
    print('Begin Execution for iteration:',counter)
    request = PublishToIoTCoreRequest()
    request.topic_name = topic
    payload = '{'+\
        '"message":"'+message + str(counter)+'",'+\
        '"clientType":"'+clientType+'"'\
              '}'
    request.payload = bytes(payload, "utf-8")    
    request.qos = qos    
    operation = ipc_client.new_publish_to_iot_core()
    operation.activate(request)    
    future_response = operation.get_response()    
    future_response.result(TIMEOUT)
    counter += 1
    time.sleep(60)
    pass
