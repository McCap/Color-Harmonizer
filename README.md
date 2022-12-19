# Color Hamronizer

## Introduction
Here is an idea for a color processing tool I had about a year ago. 
I often like to tone images based on the prevalent colors in an image. 
What I normally try is to _pull_ adjacent colors towards a selected color in order to get a more harmonious image. 
This can be done with various tools from various softwares (eg. in Rawtherapee's LAB-Tool I use the HH curve).
But it would be nice if I could select the color (the other colors are pulled towards) directly in the image.
I have written a little Python script which works on very small images, but I would like to see it usable for larger (normal) images as well.
I started to refresh my non existing C++ skills in attempt to maybe integrate it into RT someday, but this is not realistic as work and family take up to much time at the moment to do anything else with some intensity.
So I will share the idea and the existing script here to maybe inspire somebody!

## Basic Idea
The idea is to take a color space like LAB, which has a two dimensional color plain (A, B) detached from the luminosity (L). 
The idea is to select a so called _pull color_ from the image itself, and all other colors are then pulled towards the _pull color_. 
The amount of pull is defined by a __strength__. If two or more _pull colors_ are selected, then each other color is pulled towards its closest pull color. 
So how do I define a "closest" color? 
That depends on the color space. 
Ideally it is done in a color space like LAB were you have a plane that defines all the colors and is detached from the luminosity.

## Rough Script
A very rough working script in python using OpenCV is available here:
LINK
__Be warned it is very slow.__ I don't use images were the largest side is greater than 800px.
The program allows to select _pull colors_ by clicking on the image itself. It allows up to three _pull colors_.
- The first _pull color_ is selected by a left mouse click.
- The second _pull color_ is selected by a right mouse click.
- The third _pull color_ is selected by a click on the mouse wheel.
You can set the __strength__ of the pull, the __number__ of pull points and a __blur__, which I will explain later.

The program basically looks where the selected _pull color_ is located in Lab's AxB-plane an geometrically selects which colors are pulled towards which _pull color_. 


## Examples
For the examples go to the pixls.us discussion thread!


## Further ideas
- make it possible to select wether a _pull color_ will just adjust the tone of other colors while keeping it's saturation, or if it should also change the saturation
- make it possible to select the colors from a color gradient instead of selecting them from the image.
- and obviously make so that everything stays within gamut...






