import cv2
import sys
import numpy as np
#from PyQt5.QtWidgets import  QWidget, QLabel, QApplication,QPushButton,QMessageBox,QLineEdit,QErrorMessage,QMainWindow,QDesktopWidget
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import *
from  PyQt5 import uic
# from ui_form imoprt Ui_Form
stop_thread=0
rstp_client=0
start_thread=1
weightspath=None  
cfgpath=None  
classnames=None  

class Thread(QThread):
	changePixmap = pyqtSignal(QImage,int,int,int,int)

	def run(self):
		global stop_thread
		net =cv2.dnn.readNet("../yolov3.weights","../cfg/yolov3.cfg")
		with open ('coco.names','r') as f:
			classes =f.read().splitlines()
		global rstp_client
		cap=cv2.VideoCapture(rstp_client)
		#counter=0
		while True:
			if stop_thread==1:
				cap.release()
				break
			
			ret, img = cap.read()
			

			if ret:

				height,width,_=img.shape
				#img=cv2.resize(img,(height,width))
				blob=cv2.dnn.blobFromImage(img,1/255, (416,416) , (0,0,0), swapRB=True,crop=False)
				net.setInput(blob)
				output_layers_names=net.getUnconnectedOutLayersNames()
				layersOutputs=net.forward(output_layers_names)
				boxes=[]
				confidences=[]
				class_ids=[]
				for output in layersOutputs:
					for detection in output:
						scores=detection[5:]
						class_id=np.argmax(scores)
						confidence=scores[class_id]

						if confidence>0.5:
							center_x=int(detection[0]*width)
							center_y=int(detection[1]*height)
							w=int(detection[2]*width)
							h=int(detection[3]*height)
							x=int(center_x-width/4)
							y=int(center_y-height/4)
							boxes.append([x,y,w,h])
							confidences.append((float(confidence)))
							class_ids.append(class_id)
				indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
				font =cv2.FONT_HERSHEY_PLAIN
				colors=np.random.uniform(0,255,size=(len(boxes),3))
				if isinstance(indexes,np.ndarray):
					
					for i in indexes.flatten():
						x,y,w,h=boxes[i]
						label=str(classes[class_ids[i]])
						confidence=str(round(confidences[i],2))
						color=colors[i]
						cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
						cv2.putText(img,label+" "+confidence,(x,y+20),font,2,(255,255,255),2)
						rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
						rgbImage=cv2.resize(rgbImage,(800,500))
						h, w, ch = rgbImage.shape
						bytesPerLine = ch * w
						convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
						p = convertToQtFormat
						#.scaled(416, 416, Qt.KeepAspectRatio)
						self.changePixmap.emit(p,int(x),int(w),int(y),int(h))

			#cv2.imshow('tst',img)
			#counter+=1
			#if counter==30:
				#break
			#cv2.waitKey(1)
			# rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			# h, w, ch = rgbImage.shape
			# bytesPerLine = ch * w
			# convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
			# p = convertToQtFormat
			# #.scaled(416, 416, Qt.KeepAspectRatio)
			# self.changePixmap.emit(p)


class mainwindow(QMainWindow):
	def __init__(self):
		super(mainwindow,self).__init__()
		#QMainWindow.__init__(self)
		# self.title="RTCAD"
		# self.left=0
		# self.top=0
		# self.width=1000
		# self.height=620
		# self.setWindowIcon(QIcon('tst.png'))
		uic.loadUi("Bitirme2.ui",self)
		self.setWindowTitle('RTD')
		#self.ui.setupUi(self)
		self.initUI()


	@pyqtSlot(QImage,int,int,int,int)
	def setImage(self,image,x0,x1,y0,y1):
		global start_thread
		if start_thread:
			self.completed=0 
			
			while self.completed<100:
				self.completed+=0.0001
				self.progress.setValue(self.completed)
			start_thread=0
			self.progress.setVisible(False)
		
		self.label.setPixmap(QPixmap.fromImage(image))
		xcenter=(x1+x0)/2 
		ycenter=(y1+y0)/2  
		self.xpos0.setText(str(x0))
		self.xpos1.setText(str(x1+x0))
		self.ypos0.setText(str(y0))
		self.ypos1.setText(str(y1+y0))
		self.x_center.setText(str(xcenter))
		self.y_center.setText(str(ycenter))
		

	def initUI(self):
		#self.setWindowTitle(self.title)
		resolution = QDesktopWidget().screenGeometry()
		self.width=resolution.width()//2
		self.height=int(resolution.height()//3)*2
		#self.setGeometry(self.left, self.top, self.width, self.height)
		
		
		self.move((resolution.width() / 2) - (self.frameSize().width() / 2),(resolution.height() / 2) - (self.frameSize().height() / 2))
		


		#self.resize(700, 700)
		# create a label
		self.label = self.findChild(QLabel,"ImageLabel")
		self.xpos0=self.findChild(QTextBrowser,"xpos0")
		self.xpos1=self.findChild(QTextBrowser,"xpos1")
		self.ypos0=self.findChild(QTextBrowser,"ypos0")
		self.ypos1=self.findChild(QTextBrowser,"ypos1")
		self.x_center=self.findChild(QTextBrowser,"xcenter")
		self.y_center=self.findChild(QTextBrowser,"ycenter")
		#self.label.move(50, 50)
		#self.label.resize(800, 500)
		self.button=self.findChild(QPushButton,"start")
		#self.button.move(110,630)
		self.button.clicked.connect(self.startDetection)
		#self.stop_button=QPushButton('stop',self)
		#self.stop_button.move(200,630)
		#self.stop_button.clicked.connect(self.stopDetection)
		self.cam=self.findChild(QLineEdit,"camsource")
		self.Weights=self.findChild(QLineEdit,"weights")
		self.Cfg=self.findChild(QLineEdit,"cfg")
		self.classid=self.findChild(QLineEdit,"classNames")

		self.progress=self.findChild(QProgressBar,"progressBar")
		self.progress.setVisible(True)
		#self.Entry.move(200,580)
		#App.aboutToQuit.connect(self.closeGUI)
		
		self.show()
	def stopDetection(self):
		global stop_thread
		stop_thread=1
	
	def closeEvent(self,event):
		global stop_thread
		stop_thread=1
		reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
				QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

		if reply == QMessageBox.Yes:
			event.accept()
			print('Window closed')
		else:
			self.startDetection()
			event.ignore()
	def startDetection(self):

		global rstp_client
		global weightspath
		weightspath=str(self.Weights.text()) 
		global cfgpath
		cfgpath=str(self.Cfg.text())
		global classnames  
		classnames=str(self.classid.text())

		self.Weights.setVisible(False)
		self.Cfg.setVisible(False)
		self.classid.setVisible(False)  
		rstp_client=str(self.cam.text())
		self.cam.setVisible(False)
		if rstp_client=="0":
			rstp_client=int(rstp_client)
		try:
			cap = cv2.VideoCapture(rstp_client)
			if not cap.isOpened():
				QMessageBox.about(self,'Warning',"can't find cam source")

			elif cap.isOpened():
				

				global stop_thread
				stop_thread=0

				self.th = Thread(self)
				
				self.th.daemon=True
				self.th.changePixmap.connect(self.setImage)
				self.th.start()

		except:
			print("unknonw error")
		#self.show()
if __name__=="__main__":
	App=QApplication(sys.argv)

	window=mainwindow()
	sys.exit(App.exec())