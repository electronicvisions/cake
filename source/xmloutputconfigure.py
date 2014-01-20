###################################################################
#
#                 XML configuration file creation
#
# Company         :   KIP
# Author          :   Marc-Olivier Schwartz
# E-Mail          :   marcolivier.schwartz@kip.uni-heidelberg.de
#
###################################################################

# Import

from lxml import etree
import os

# Main class

class XMLOutput:

	def __init__ (self,xmlfile_path):
		self.xmlfile_path = xmlfile_path

		self.denVals = (500,0,0,0,0,0,0,0)
		self.writeVals = (255,2,16,63,4,8,15,0)

	def create_xml(self,pArrayTopLeft,pArrayTopRight,pArrayBotLeft,pArrayBotRight,pGlobal,n_index_TopLeft,n_index_TopRight,n_index_BotLeft,n_index_BotRight):
      
		reticle = etree.Element("reticle") 			# Creating reticle

		type = etree.SubElement(reticle,"type")
		hicann = etree.SubElement(reticle,"hicann", id = "0")	# Creating HICANN
      
		output = etree.SubElement(hicann, "output")
      
		index = range(2)
		for i in index:
			op = etree.SubElement(output, "op", id = str(i), state = str(1))
		op.text = "0"

		neurons = etree.SubElement(hicann, "neurons")

		denmem = etree.SubElement(neurons, "denmem", id = "0")

		El = etree.SubElement(denmem, "El")
		El.text = str(self.denVals[0])

		Vt = etree.SubElement(denmem, "Vt")
		Vt.text = str(self.denVals[1])	

		Vsyni = etree.SubElement(denmem, "Vsyni")
		Vsyni.text = str(self.denVals[2])

		Vsynx = etree.SubElement(denmem, "Vsynx")
		Vsynx.text = str(self.denVals[3])

		Vsyntci = etree.SubElement(denmem, "Vsyntci")
		Vsyntci.text = str(self.denVals[4])

		Vsyntcx = etree.SubElement(denmem, "Vsyntcx")
		Vsyntcx.text = str(self.denVals[5])

		Esyni = etree.SubElement(denmem, "Esyni")
		Esyni.text = str(self.denVals[6])

		Esynx = etree.SubElement(denmem, "Esynx")
		Esynx.text = str(self.denVals[7])
      
      	# Only one array
		if (pArrayTopLeft != [] and pArrayTopRight == []):
			FGS = ['0','1','all']
			for item in FGS:
				self.create_array(hicann,item,n_index_TopLeft,pArrayTopLeft,pGlobal)

		if (pArrayTopLeft == [] and pArrayTopRight != []):
			FGS = ['0','1','all']
			for item in FGS:
				self.create_array(hicann,item,n_index_TopRight,pArrayTopRight,pGlobal)

		if (pArrayBotLeft == [] and pArrayBotRight != []):
			FGS = ['2','3','all']
			for item in FGS:
				self.create_array(hicann,item,n_index_BotRight,pArrayBotRight,pGlobal)

		if (pArrayBotLeft != [] and pArrayBotRight == []):
			FGS = ['2','3','all']
			for item in FGS:
				self.create_array(hicann,item,n_index_BotLeft,pArrayBotLeft,pGlobal)

		# 2 FG arrays or all FG arrays
		if (pArrayTopLeft != [] and pArrayTopRight != []):
			FGS = ['0']
			for item in FGS:
				self.create_array(hicann,item,n_index_TopLeft,pArrayTopLeft,pGlobal)
			FGS = ['1','all']
			for item in FGS:
				self.create_array(hicann,item,n_index_TopRight,pArrayTopRight,pGlobal)

		if (pArrayBotLeft != [] and pArrayBotRight != []):
			FGS = ['2']
			for item in FGS:
				self.create_array(hicann,item,n_index_BotLeft,pArrayBotLeft,pGlobal)
			FGS = ['3','all']
			for item in FGS:
				self.create_array(hicann,item,n_index_BotRight,pArrayBotRight,pGlobal)
		 
		# Other lines
		stimulus = etree.SubElement(hicann, "stimulus", id = "all")
		stimulus.text = ""

		#print etree.tostring(reticle, pretty_print=True)
      
		result = etree.tostring(reticle, pretty_print=True)
      
		# Add DOCTYPE
		result = "<!DOCTYPE stage2conf>" + "\n" + result

		xmlfile = os.path.join(self.xmlfile_path, "FGparam.xml")

		f = open(xmlfile, "w") # Writing XML file
		#print "XML file written to", xmlfile
		f.write(str(result))
		f.close()


	# Create an FG array
	def create_array(self,hicann,fg_id,index,array,global_array):

		fg = etree.SubElement(hicann, "fg", id = fg_id)
		  
		maxcycle = etree.SubElement(fg, "maxcycle")
		maxcycle.text = str(self.writeVals[0])

		currentwritetime = etree.SubElement(fg, "currentwritetime")

		currentwritetime.text = str(self.writeVals[1])
		voltagewritetime = etree.SubElement(fg, "voltagewritetime")
		voltagewritetime.text = str(self.writeVals[2])

		readtime = etree.SubElement(fg, "readtime")
		readtime.text = str(self.writeVals[3])
		
		acceleratorstep = etree.SubElement(fg, "acceleratorstep")
		acceleratorstep.text = str(self.writeVals[4])

		fg_pulselength = etree.SubElement(fg, "fg_pulselength")
		fg_pulselength.text = str(self.writeVals[5])

		fg_bias = etree.SubElement(fg, "fg_bias")
		fg_bias.text = str(self.writeVals[6])

		fg_biasn = etree.SubElement(fg, "fg_biasn")
		fg_biasn.text = str(self.writeVals[7])

		indexLine = range(24)					# Creating lines
		for j in indexLine:
			line = etree.SubElement(fg, "line", id = str(j))
			valueall = etree.SubElement(line, "value", id = "all")
			valueall.text = str(array[0][j])
			valuezero = etree.SubElement(line, "value", id = "0")
			valuezero.text = str(global_array[j])
				
			for k, item in enumerate(array):
				valueneuron = etree.SubElement(line, "value", id = str(index[k]+1))
				valueneuron.text = str(item[j])
		line = etree.SubElement(fg, "line", id = "all")
		valueall = etree.SubElement(line, "value", id = "all")
		valueall.text = "0"

