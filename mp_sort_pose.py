

class SortPose:
	# """Sort image files based on head pose"""

	def __init__(self, motion):
		if motion['side_to_side'] == True:
			self.XLOW = -20
			self.XHIGH = 1
			self.YLOW = -30
			self.YHIGH = 30
			self.ZLOW = -1
			self.ZHIGH = 1
			self.MINCROP = 1
			self.MAXRESIZE = .5
			self.MAXMOUTHGAP = 4
			self.FRAMERATE = 15
			self.SORT = 'y'
			self.SECOND_SORT = 'x'
			# self.SORT = 'mouth_gap'
			self.ROUND = 0
		elif motion['forward_smile'] == True:
			self.XLOW = -20
			self.XHIGH = 1
			self.YLOW = -4
			self.YHIGH = 4
			self.ZLOW = -3
			self.ZHIGH = 3
			self.MINCROP = 1
			self.MAXRESIZE = .5
			self.FRAMERATE = 15
			self.SECOND_SORT = 'x'
			# self.# MAXMOUTHGAP = 40
			self.SORT = 'mouth_gap'
			self.ROUND = 1
		elif motion['forward_nosmile'] == True:
			self.XLOW = -20
			self.XHIGH = 1
			self.YLOW = -4
			self.YHIGH = 4
			self.ZLOW = -3
			self.ZHIGH = 3
			self.MINCROP = 1
			self.MAXRESIZE = .5
			self.FRAMERATE = 15
			self.SECOND_SORT = 'x'
			self.MAXMOUTHGAP = 2
			self.SORT = 'mouth_gap'
			self.ROUND = 1
		elif motion['static_pose'] == True:
			self.XLOW = -20
			self.XHIGH = 1
			self.YLOW = -4
			self.YHIGH = 4
			self.ZLOW = -3
			self.ZHIGH = 3
			self.MINCROP = 1
			self.MAXRESIZE = .5
			self.FRAMERATE = 15
			self.SECOND_SORT = 'mouth_gap'
			self.MAXMOUTHGAP = 10
			self.SORT = 'x'
			self.ROUND = 1
		elif motion['simple'] == True:
			self.XLOW = -20
			self.XHIGH = 1
			self.YLOW = -4
			self.YHIGH = 4
			self.ZLOW = -3
			self.ZHIGH = 3
			self.MINCROP = 1
			self.MAXRESIZE = .5
			self.FRAMERATE = 15
			self.SECOND_SORT = 'mouth_gap'
			self.MAXMOUTHGAP = 10
			self.SORT = 'x'
			self.ROUND = 1