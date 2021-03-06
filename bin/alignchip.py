#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2017 by Sven Kochmann
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# Description:
# 1.3 Alignment setup (GUI program)
#
# This program helps the user to align the chip using a grid and overlapping rectangles in order to optimize the
# visual representation of the measurement for both the user and the evaluation scripts. This program used the
# settings from the first step (1.1).
#

# Import modules
import wx               # wxPython (wxWidgets)
import cv2              # OpenCV
import numpy as np      # Numpy - You always need this.
import ConfigParser     # Reading/Writing config files
import ffe              # frequently-used function script
import time             # Time functions

# Main dialog
class MyDialog(wx.Dialog):
    # Init function (onInitDialog)
    def __init__(self, parent):
        # This layout code was generated by wxFormBuilder
        # -----------------------------------------------
        wx.Dialog.__init__(self, parent, id=wx.ID_ANY, title=u"Alignment setup", pos=wx.DefaultPosition,
                           size=wx.Size(1000, 609), style=wx.DEFAULT_DIALOG_STYLE)

        self.SetSizeHintsSz(wx.DefaultSize, wx.DefaultSize)

        bSizer1 = wx.BoxSizer(wx.HORIZONTAL)

        self.campanel = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.Size(640, 480), wx.TAB_TRAVERSAL)
        bSizer1.Add(self.campanel, 1, wx.ALL, 5)

        bSizer2 = wx.BoxSizer(wx.VERTICAL)

        self.navpanel = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.Size(256, 192), wx.TAB_TRAVERSAL)
        self.navpanel.SetMaxSize(wx.Size(256, 192))

        bSizer2.Add(self.navpanel, 1, wx.ALL, 5)

        gSizer1 = wx.GridSizer(0, 2, 0, 0)

        self.toggle_view = wx.ToggleButton(self, wx.ID_ANY, u"Toggle view", wx.DefaultPosition, wx.DefaultSize, 0)
        gSizer1.Add(self.toggle_view, 0, wx.ALL, 5)

        gSizer1.AddSpacer((0, 0), 1, wx.EXPAND, 5)

        self.m_staticline2 = wx.StaticLine(self, wx.ID_ANY, wx.DefaultPosition, wx.Size(100, -1), wx.LI_HORIZONTAL)
        gSizer1.Add(self.m_staticline2, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        gSizer1.AddSpacer((0, 0), 1, wx.EXPAND, 5)

        self.static_grid = wx.StaticText(self, wx.ID_ANY, u"Grid options", wx.DefaultPosition, wx.DefaultSize, 0)
        self.static_grid.Wrap(-1)
        gSizer1.Add(self.static_grid, 0, wx.ALL, 5)

        gSizer1.AddSpacer((0, 0), 1, wx.EXPAND, 5)

        self.static_gridsize = wx.StaticText(self, wx.ID_ANY, u"Grid size:", wx.DefaultPosition, wx.DefaultSize, 0)
        self.static_gridsize.Wrap(-1)
        gSizer1.Add(self.static_gridsize, 0, wx.ALL, 5)

        self.spin_gridsize = wx.SpinCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize,
                                         wx.SP_ARROW_KEYS | wx.TE_PROCESS_ENTER, 10, 200, 0)
        gSizer1.Add(self.spin_gridsize, 0, wx.ALL, 5)

        self.static_gridcolor = wx.StaticText(self, wx.ID_ANY, u"Grid color:", wx.DefaultPosition, wx.DefaultSize, 0)
        self.static_gridcolor.Wrap(-1)
        gSizer1.Add(self.static_gridcolor, 0, wx.ALL, 5)

        self.color_grid = wx.ColourPickerCtrl(self, wx.ID_ANY, wx.Colour(255, 255, 0), wx.DefaultPosition,
                                              wx.DefaultSize, wx.CLRP_DEFAULT_STYLE)
        gSizer1.Add(self.color_grid, 0, wx.ALL, 5)

        self.m_staticline1 = wx.StaticLine(self, wx.ID_ANY, wx.DefaultPosition, wx.Size(100, -1), wx.LI_HORIZONTAL)
        gSizer1.Add(self.m_staticline1, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        gSizer1.AddSpacer((0, 0), 1, wx.EXPAND, 5)

        self.static_rectangle = wx.StaticText(self, wx.ID_ANY, u"Rectangle options", wx.DefaultPosition, wx.DefaultSize,
                                              0)
        self.static_rectangle.Wrap(-1)
        gSizer1.Add(self.static_rectangle, 0, wx.ALL, 5)

        gSizer1.AddSpacer((0, 0), 1, wx.EXPAND, 5)

        self.static_threshhold1 = wx.StaticText(self, wx.ID_ANY, u"Threshhold (bifilter):", wx.DefaultPosition,
                                                wx.DefaultSize, 0)
        self.static_threshhold1.Wrap(-1)
        gSizer1.Add(self.static_threshhold1, 0, wx.ALL, 5)

        self.spin_threshhold1 = wx.SpinCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize,
                                            wx.SP_ARROW_KEYS | wx.TE_PROCESS_ENTER, 0, 255, 0)
        gSizer1.Add(self.spin_threshhold1, 0, wx.ALL, 5)

        self.static_threshhold2 = wx.StaticText(self, wx.ID_ANY, u"Threshhold (binary):", wx.DefaultPosition,
                                                wx.DefaultSize, 0)
        self.static_threshhold2.Wrap(-1)
        gSizer1.Add(self.static_threshhold2, 0, wx.ALL, 5)

        self.spin_threshhold2 = wx.SpinCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize,
                                            wx.SP_ARROW_KEYS | wx.TE_PROCESS_ENTER, 0, 255, 0)
        gSizer1.Add(self.spin_threshhold2, 0, wx.ALL, 5)

        bSizer2.Add(gSizer1, 1, wx.ALIGN_TOP, 5)

        bSizer1.Add(bSizer2, 1, 0, 5)

        self.SetSizer(bSizer1)
        self.Layout()

        self.Centre(wx.BOTH)

        # Connect Events
        self.Bind(wx.EVT_CLOSE, self.onClose)
        self.campanel.Bind(wx.EVT_ERASE_BACKGROUND, self.onEraseBackground)
        self.campanel.Bind(wx.EVT_PAINT, self.onPaintMain)
        self.navpanel.Bind(wx.EVT_ERASE_BACKGROUND, self.onEraseBackground)
        self.navpanel.Bind(wx.EVT_PAINT, self.onPaintNav)
        self.spin_gridsize.Bind(wx.EVT_TEXT_ENTER, self.onUpdateSettings)
        self.color_grid.Bind(wx.EVT_COLOURPICKER_CHANGED, self.onUpdateSettings)
        self.spin_threshhold1.Bind(wx.EVT_TEXT_ENTER, self.onUpdateSettings)
        self.spin_threshhold2.Bind(wx.EVT_TEXT_ENTER, self.onUpdateSettings)
        # -----------------------------------------------

        # Initialize camera
        self.cam = cv2.VideoCapture(ffe.getCameraID())

        # Apply camera settings; load recording settings
        self.camsettings = ffe.applyCameraSettings(self.cam)

        ffe.dumpCameraPropsToConsole(self.cam)

        # Get camera width and height
        self.framewidth = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameheight = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Try to get very first frame from camera
        ret, image = self.cam.read()

        # Successful? (i.e. is there a camera?)
        if ret == True:
            # Create grid
            self.grid_size = 20
            self.grid_color = (0, 255, 255)
            self.spin_gridsize.Value = self.grid_size
            self.color_grid.Colour = self.grid_color[::-1]
            image_grid = self.createGrid(image)

            # Create rectangles
            self.threshhold1 = 70
            self.threshhold2 = 100
            self.spin_threshhold1.Value = self.threshhold1
            self.spin_threshhold2.Value = self.threshhold2
            image_rect = self.createRectangles(image)

            # Cameras like Blue-green-red, but panels like Red-green-blue, so we have to convert
            image_grid = cv2.cvtColor(image_grid, cv2.COLOR_BGR2RGB)
            image_rect = cv2.cvtColor(image_rect, cv2.COLOR_BGR2RGB)

            # Get dimensions of images and copy to bitmap
            row, col, c = image_grid.shape
            self.mainbmp = wx.BitmapFromBuffer(col, row, image_grid)
            row, col, c = image_rect.shape
            self.navbmp = wx.BitmapFromBuffer(col, row, image_rect)

            # Create timer for frame grabbing
            self.playTimer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self.onNextFrame)
            self.playTimer.Start(50)

            # Create empty array for integrated frames
            self.lastframe = np.zeros((self.frameheight, self.framewidth), np.uint32)  # 32-bit Integer!
            self.showframe = np.zeros((self.frameheight, self.framewidth), np.uint8)
            self.latesttime = time.clock()

        # No camera? Get out of here
        else:
            # Show message to user
            wx.MessageBox("Error reading frames from camera. Is the camera connected?", "Camera settings", wx.ICON_STOP)
            # Destroy dialog
            self.EndModal(wx.ID_ABORT)

    # onClose: Release camera, kill timer, destroy window
    def onClose(self, evt):
        # Stop timer
        self.playTimer.Stop()
        # Release camera
        self.cam.release()
        # Destroy window
        self.Destroy()

    # onEraseBackground: Do nothing. Absolutely nothing.
    def onEraseBackground(self, evt):
        pass

    # onPaint
    def onPaintMain(self, evt):
        # If there is a bitmap to draw
        if self.mainbmp:
            # Paint it on myself (i.e. the panel)
            dc = wx.BufferedPaintDC(self.campanel)
            dc.DrawBitmap(self.mainbmp, 0, 0, True)
        # Skip
        evt.Skip()

    # onPaint
    def onPaintNav(self, evt):
        # If there is a bitmap to draw
        if self.navbmp:
            # Create image from bitmap
            img = self.navbmp.ConvertToImage()
            img.Rescale(self.navpanel.Size[0], self.navpanel.Size[1])
            scaledbmp = img.ConvertToBitmap()

            # Paint it on myself (i.e. the panel)
            dc = wx.BufferedPaintDC(self.navpanel)
            dc.DrawBitmap(scaledbmp, 0, 0, True)
        # Skip
        evt.Skip()

    # onTimer
    def onNextFrame(self, evt):
        # Read a frame
        ret, image = self.cam.read()
        # Successful?
        if ret == True:
            # Flipping?
            if self.camsettings["doflipping"]:
                img = cv2.flip(image, self.camsettings["flip"])

            # Splitting frame into channels
            blue, green, red = cv2.split(image)

            # Channels
            if "blue" in self.camsettings:
                self.lastframe = self.lastframe + np.array(blue, dtype=np.uint32)
            if "green" in self.camsettings:
                self.lastframe = self.lastframe + np.array(green, dtype=np.uint32)
            if "red" in self.camsettings:
                self.lastframe = self.lastframe + np.array(red, dtype=np.uint32)

            # Elapsed time
            elapsed = time.clock() - self.latesttime

            # Have reached the integrated frames?
            if elapsed > self.camsettings["integratetime"]:
                if self.lastframe.max() > 0:
                    self.lastframe = np.multiply(self.lastframe, 255) / (self.lastframe.max())
                self.showframe = np.array(self.lastframe, dtype=np.uint8)
                self.lastframe = np.zeros((self.frameheight, self.framewidth), np.uint32)
                self.latesttime = time.clock()

                colorframe = np.zeros((self.frameheight, self.framewidth, 3), dtype=np.uint8)
                colorframe[:, :, 1] = self.showframe

                # Based on this frame, we create the grid and rectangle
                image_grid = self.createGrid(colorframe)
                image_rect = self.createRectangles(colorframe)

                # Cameras like Blue-green-red, but panels like Red-green-blue, so we have to convert
                image_grid = cv2.cvtColor(image_grid, cv2.COLOR_BGR2RGB)
                image_rect = cv2.cvtColor(image_rect, cv2.COLOR_BGR2RGB)

                # Get dimensions of images and copy to bitmap
                row, col, c = image_grid.shape
                self.mainbmp = wx.BitmapFromBuffer(col, row, image_grid)
                row, col, c = image_rect.shape
                self.navbmp = wx.BitmapFromBuffer(col, row, image_rect)

                # Swap if button is pressed
                if self.toggle_view.Value == 1:
                    self.mainbmp, self.navbmp = self.navbmp, self.mainbmp

                self.campanel.Refresh()
                self.navpanel.Refresh()
        evt.Skip()

    # onUpdateSettings
    def onUpdateSettings(self, evt):
        # Get the settings from the controls
        self.grid_size = self.spin_gridsize.Value
        self.grid_color = self.color_grid.Colour[::-1]
        self.threshhold1 = self.spin_threshhold1.Value
        self.threshhold2 = self.spin_threshhold2.Value
        evt.Skip()

    def createGrid(self, frame):
        # Getting the shape of the image
        height, width, bits = frame.shape

        # Create empty grid frame
        image_grid = np.zeros((height, width, bits), np.uint8)

        # Drawing the grid on the copy of the frame
        n_lines_x = int(width / self.grid_size)
        n_lines_y = int(height / self.grid_size)

        for x in xrange(n_lines_x):
            cv2.line(image_grid, ((x + 1) * self.grid_size, 0), ((x + 1) * self.grid_size, height), self.grid_color, 1, 8, 0)

        for y in xrange(n_lines_y):
            cv2.line(image_grid, (0, (y + 1) * self.grid_size), (width, (y + 1) * self.grid_size), self.grid_color, 1, 8, 0)

        # Add grid to frame
        frame = cv2.addWeighted(image_grid, 0.5, frame, 1.0, 0)

        # Return frame
        return frame

    def createRectangles(self, frame):
        # Getting the shape of the image
        height, width, bits = frame.shape

        # Splitting channels
        frame_blue, frame_green, frame_red = cv2.split(frame)

        # Applying the bilateralFilter
        frame_bifilter = cv2.bilateralFilter(frame_green, 17, self.threshhold1, 17)

        # Threshhold function
        ret, frame_thresh = cv2.threshold(frame_bifilter, self.threshhold2, 255, cv2.THRESH_BINARY)

        # Creating two blank images for the rectangles (filled)
        frame_rectangle1 = np.zeros((height, width, bits), np.uint8)
        frame_rectangle2 = np.zeros((height, width, bits), np.uint8)

        # Finding the contours
        contour_image, contours, hierarchy = cv2.findContours(frame_thresh, 1, 2)

        # No contours? Return empty image
        if len(contours) == 0:
            return frame_rectangle2

        # Select the contour with the largest area
        counter_sel = 0
        area = cv2.contourArea(contours[counter_sel])
        for i in xrange(len(contours)):
            if cv2.contourArea(contours[i]) > area:
                counter_sel = i
                area = cv2.contourArea(contours[counter_sel])

        # Calculating the bounding rectangle (not minimal)
        x, y, w, h = cv2.boundingRect(contours[counter_sel])
        frame_rectangle1 = cv2.rectangle(frame_rectangle1, (x, y), (x + w, y + h), (0, 255, 0), -1)

        # Calculating the MINIMAL bounding rectangle
        rect = cv2.minAreaRect(contours[counter_sel])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        frame_rectangle2 = cv2.drawContours(frame_rectangle2, [box], 0, (0, 0, 255), -1)

        # Combine both rectangles
        frame_rectangle2 = frame_rectangle1 + frame_rectangle2

        # Count yellow pixels
        dest = cv2.inRange(frame_rectangle2, (0, 255, 255), (0, 255, 255))
        yellow_pixels = cv2.countNonZero(dest)

        # Count green pixels
        dest = cv2.inRange(frame_rectangle1, (0, 255, 0), (0, 255, 0))
        green_pixels = cv2.countNonZero(dest)

        # Calculate and format ratio
        ratio = '%.1f%%' % (float(yellow_pixels)/float(green_pixels)*100.0)

        # Add text to bounding-rectangle-frame for the overlap
        size, baseline = cv2.getTextSize(ratio, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        textwidth, textheight = size
        cv2.putText(frame_rectangle2, ratio, (10, textheight + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, 8, False)

        # Return image
        return frame_rectangle2



# Actual main program: Show the above dialog
if __name__=="__main__":
    app = wx.App()
    dlg = MyDialog(None)
    dlg.ShowModal()
    app.ExitMainLoop()
