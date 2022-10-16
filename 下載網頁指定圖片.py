##下載網頁上指定的圖片
##僅限一般下載，未對網頁本身對圖片的保護有任何突破
##介面、測試完成

import requests, os
from copy import deepcopy

saveDir = input('載完放哪?')

if not os.path.isdir(saveDir):
    os.mkdir(saveDir)

targetUrl = input('目標圖在哪?(右鍵複製圖片位址)')
many = str(input('執行次數?(起始值 次數)')).split(' ')

if len(many) == 1:
    tmp = deepcopy(many)
    many[0] = '0'
    many.extend(tmp)

print('\n開始下載...\n')

try:
    for num in range(int(many[1])):
        img = requests.get(targetUrl)
        with open(saveDir + os.sep + str(num + int(many[0])) + '.jpg', 'wb') as f:
            f.write(img.content)
        
        print('- 完成下載 ' + saveDir + os.sep + str(num + int(many[0])) + '.jpg')
except:
    print('不明的程式中斷')
    traceback.print_exc()
    os.system('pause')
    
print('完成作業\n')
os.system('pause')
