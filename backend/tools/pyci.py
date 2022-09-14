"""
pyci.py
Python bindings for CoreImage
Copyright 2018 Apple Inc. All rights reserved.
# Install
    1. pip install pyobjc --ignore-installed --user
    2. pip install numpy --ignore-installed --user
"""

""" Flags """
FLAG_DEBUG = False

""" Standard imports """
from warnings import warn
import inspect
from difflib import get_close_matches
import os

""" Frameworks """
import numpy as np
from numpy import ndarray

""" Foundation """
from Foundation import NSData, NSURL, NSMutableData

""" Core Graphics """
from Quartz import CoreGraphics
from Quartz import CGDataProviderCreateWithCFData, CGImageCreate
from Quartz import CGColorSpaceCreateWithName, CGColorSpaceCreateDeviceGray
from Quartz import CGAffineTransform, CGAffineTransformMake, CGAffineTransformMakeScale, \
    CGAffineTransformMakeTranslation, CGAffineTransformMakeRotation
from Quartz import CGRect, CGSize, CGPoint
from Quartz import CGRectMake, CGRectInset, CGRectIntegral

""" Core Image """
from Quartz import CIImage, CIFilter, CIContext, CIColor, CIVector, CIKernel, CIColorKernel

""" Constants """
from Quartz import kCGBitmapByteOrderDefault, kCGRenderingIntentDefault
from Quartz import kCGBitmapFloatComponents, kCGBitmapByteOrder32Little, kCGBitmapByteOrder16Little
from Quartz import kCGColorSpaceSRGB, kCGColorSpaceLinearSRGB

# Enumerate missing keys
kCIFormatRGBA8 = 24
kCIFormatRGBAh = 31
kCIFormatRGBAf = 34
# kCIFormatRGB8 = 19
# kCIFormatRGBh = 30
# kCIFormatRGBf = 32

kCIContextOutputColorSpace = 'output_color_space'
kCIContextWorkingColorSpace = 'working_color_space'
kCIContextWorkingFormat = 'working_format'
kCIContextUseSoftwareRenderer = 'software_renderer'
kCIContextQuality = 'quality'
kCIContextHighQualityDownsample = 'high_quality_downsample'
kCIContextOutputPremultiplied = 'output_premultiplied'
kCIContextCacheIntermediates = 'kCIContextCacheIntermediates'
kCIActiveKeys = 'activeKeys'


def show(img, title=None, color='gray', at=None, interpolation='bilinear', forced=False, sub=False):
    """ matplotlib.imshow helper """
    import matplotlib
    import matplotlib.pylab as plt
    matplotlib.pylab.rcParams['figure.figsize'] = (18.0, 18 / 1.6)

    if isinstance(img, (tuple, list)):
        n = len(img)
        for i in range(n):
            t = title[i] if isinstance(title, (tuple, list)) else title
            show(img[i], title=t, at='1{}{}'.format(n, i + 1), color=color, interpolation=interpolation, forced=forced,
                 sub=True)
        if 'inline' not in matplotlib.get_backend():
            plt.show()
        return

    if isinstance(img, cimg):
        img = img.render()

    if at:
        if isinstance(at, (int)):
            plt.subplot(at)
        elif isinstance(at, (tuple, list)) and len(at) == 3:
            plt.subplot(*at)
        else:
            plt.subplot(at)

    if img is not None:
        plt.imshow(img.clip(0, 1), interpolation=interpolation, cmap=plt.get_cmap(color))

    if title:
        plt.title(title, color='w')

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    last = False
    if isinstance(at, (int)):
        last = at and int(str(at)[2]) == int(str(at)[0]) * int(str(at)[1])
    elif isinstance(at, (list, tuple)) and len(at) == 3:
        last = at[2] == at[0] * at[1]

    if (forced or 'inline' not in matplotlib.get_backend()) and not sub and not at or last:
        import matplotlib.pylab
        plt.show()


""" Defaults """
DEFAULT_FORMAT_RENDER = np.float32  # render from CIImage to numpy
DEFAULT_FORMAT_WORKING = np.float32  # intermediates
DEFAULT_FORMAT_FROMFILE = np.float32
DEFAULT_COLORSPACE = kCGColorSpaceSRGB

""" Conversion helpers """
_pixelFormatsRGBA = (np.float16, kCIFormatRGBAh), (np.float32, kCIFormatRGBAf), (np.uint8, kCIFormatRGBA8)
dtype2formatRGBA = {d: f for (d, f) in _pixelFormatsRGBA}
format2dtypeRGBA = {f: d for (d, f) in _pixelFormatsRGBA}
SUPPORTED_DTYPES = [dtype for (dtype, format) in _pixelFormatsRGBA]

""" Module Singleton Accessors """


def setContext(context):
    global singletonContext
    singletonContext = context


def setColorspace(colorspace):
    global singletonColorspace
    singletonColorspace = colorspace


def createContext(useDefaults=False, colorspaceWorking=None, colorspaceOutput=None, workingFormat=None, linear=False):
    """
    Args:
        colorspaceWorking (CGColorSpace): Working colorspace.
        colorspaceOutput (CGColorSpace): Output colorspace.
        workingFormat (numpy.dtype): Working format.
    Note:
        kCIContextOutputColorSpace = 'output_color_space'
        kCIContextWorkingColorSpace = 'working_color_space'
        kCIContextWorkingFormat = 'working_format'
        kCIContextUseSoftwareRenderer = 'software_renderer'
        kCIContextQuality = 'quality'
        kCIContextHighQualityDownsample = 'high_quality_downsample'
        kCIContextOutputPremultiplied = 'output_premultiplied'
        kCIContextCacheIntermediates = 'kCIContextCacheIntermediates'xw
    """

    options = {}
    options[kCIContextOutputColorSpace] = colorspaceOutput
    options[kCIContextWorkingColorSpace] = colorspaceWorking
    options[kCIContextWorkingFormat] = dtype2formatRGBA[
        workingFormat] if workingFormat in SUPPORTED_DTYPES else workingFormat

    # Prune empty keys
    empty = [option for option in options if option is None]
    for option in empty:
        options.pop(option)

    # Default override
    if useDefaults:
        options = {
            kCIContextWorkingFormat: dtype2formatRGBA[DEFAULT_FORMAT_WORKING],
            kCIContextWorkingColorSpace: singletonColorspace
        }

    if linear:
        options[kCIContextWorkingColorSpace] = CGColorSpaceCreateWithName(kCGColorSpaceLinearSRGB)
        options[kCIContextOutputColorSpace] = CGColorSpaceCreateWithName(kCGColorSpaceLinearSRGB)

    context = CIContext.contextWithOptions_(options)

    if not context:
        raise RuntimeError('unable to create context')

    setContext(context)


def createColorspace(name=DEFAULT_COLORSPACE):
    colorspace = CGColorSpaceCreateWithName(name)

    if not colorspace:
        raise RuntimeError('unable to create colorspace')

    setColorspace(colorspace)


# Create Module Singletons
createColorspace()
createContext(useDefaults=True)


# Utilities
def set_print_tree(level, more=None):
    if level < 0 or level > 8:
        warn('CI_PRINT_TREE should be in the range (0,8)')
        return

    os.environ['CI_PRINT_TREE'] = str(level)


def get_print_tree():
    os.environ['CI_PRINT_TREE']


def vector(*args):
    # Support either Vector(1, 1, 1, 1) or Vector([1, 1, 1, 1])
    lst = args if len(args) > 1 else args[0]

    if not isinstance(lst, (list, tuple)):
        raise RuntimeError('Can only create a CIVector from tuple types.')

    length = len(lst)
    if length > 4:
        raise RuntimeError('Can only create a CIVector from tuples with less than or equal to 4 components.')
    elif length == 1:
        return CIVector.vectorWithX_(*lst)
    elif length == 2:
        return CIVector.vectorWithX_Y_(*lst)
    elif length == 3:
        return CIVector.vectorWithX_Y_Z_(*lst)
    elif length == 4:
        return CIVector.vectorWithX_Y_Z_W_(*lst)


def color(*args):
    # Support either Color(1, 1, 1, X) or Color([1, 1, 1, X])
    lst = args if len(args) > 1 else args[0]

    if not isinstance(lst, (list, tuple)):
        raise RuntimeError('Can only create a CIColor from tuple types.')

    length = len(lst)

    if length == 3:
        return CIColor.colorWithRed_green_blue_(*lst)
    elif length == 4:
        return CIColor.colorWithRed_green_blue_alpha_(*lst)
    else:
        raise RuntimeError('Can only create a CIColor from tuples with 3 or 4 components but received {}.'.format(args))


def inset(rect, dx, dy, integral=True):
    if isinstance(rect, (list, tuple)):
        rect = CGRectMake(*rect)

    rect = CGRectInset(rect, dx, dy)

    if integral:
        rect = CGRectIntegral(rect)

    return rect.origin.x, rect.origin.y, rect.size.width, rect.size.height


def rectify(rect):
    if isinstance(rect, (list, tuple)):
        rect = CGRectMake(*rect)
    return rect


class cimg:
    DEFAULT_RENDER_DTYPE = np.float32
    _kernelCounter = 0

    @staticmethod
    def filters():
        return CIFilter.filterNamesInCategory_(None)

    @staticmethod
    def inputs(filterName):
        if not filterName.startswith('CI'):
            filterName = 'CI' + filterName[0].upper() + filterName[1:]
        return CIFilter.filterWithName_(filterName).attributes()

    def __init__(self, data, dtype=None):

        self.dtype = dtype if dtype else DEFAULT_FORMAT_RENDER

        if isinstance(data, cimg):
            # copy constructor
            self._cake = data._cake
            self._dirty = data._dirty
            self._ciimage = data._ciimage

        elif isinstance(data, ndarray):
            # numpy array
            # FIXME: might need to enforce byte ordering
            # data = data.astype(data.dtype, order='C', casting='unsafe', subok=True, copy=True)

            if data.ndim == 2:
                data = data[..., np.newaxis]

            if data.ndim != 3 or (data.ndim == 3 and data.shape[2] == 2):
                raise RuntimeError(
                    'Only single channel, RGB and RGBA images supported but received {}'.format(data.shape))

            if dtype:
                data = data.astype(dtype)

            self._cake = data
            self._dirty = False
            self._ciimage = cimg.create_ciimage_from_numpy(data)

        elif isinstance(data, CIImage):
            # native CIImage
            self._cake = None
            self._dirty = True
            self._ciimage = data

        else:
            raise NotImplementedError('Constructor not supported: {}'.format(type(data)))

        # Add filter lambdas
        if not getattr(self, '_lambdified', False):
            self._add_filter_lambdas()
            self._add_imageby_lambdas()
            setattr(self, '_lambdified', True)

    def __str__(self):
        return '<{}.{} at {}> extent={} dtype={}'.format(self.__class__.__module__,
                                                         self.__class__.__name__,
                                                         hex(id(self)),
                                                         self.extent,
                                                         self.dtype)

    def __getitem__(self, index):
        """ numpy style slicing. """
        # Only fully index, non-stepping slicing supported with cropping
        pre_crop = (isinstance(index, tuple) and len(index) == 3
                    and (index[0] == Ellipsis or index[0].step is None)
                    and (index[1] == Ellipsis or index[1].step is None)
                    and (index[2] == Ellipsis or index[2].step is None))

        if pre_crop:
            rows, cols, dims = index
            w, h = self.size

            x0 = cols.start if cols.start else 0
            x1 = cols.stop if cols.stop else w

            y0 = h - rows.stop if rows.stop else 0
            y1 = h - rows.start if rows.start else h

            rect = x0, y0, x1 - x0, y1 - y0
            crop = self.crop(*rect)
            ary = crop.render()
            return ary[..., index[2]] if index[2] is not Ellipsis else ary[...]

        else:
            return self.render()[index]

    def __setitem__(self, idx, value):
        # TODO: create a recipe instead of baking the array
        ary = self.render()
        ary[idx] = value
        self._cake = ary
        self._dirty = False
        self._ciimage = cimg.create_ciimage_from_numpy(ary)

    def _add_filter_lambdas(self):

        # Generate lambdas for all filters here, so that we can call them directly on the array.
        for filterName in cimg.filters():
            # Skip CICrop, will handle it as an attribute
            if filterName == 'CICrop':
                continue

            # 1) CIFilterName()
            if not hasattr(cimg, filterName):
                setattr(cimg, filterName, lambda self, foo=filterName, **kwargs: self.applyFilter(foo, **kwargs))

            # 2) FilterName()
            filterName = filterName[2:]
            setattr(cimg, filterName, lambda self, foo=filterName, **kwargs: self.applyFilter(foo, **kwargs))

            # 3) filterName()
            filterName = filterName[0].lower() + filterName[1:]
            setattr(cimg, filterName, lambda self, foo=filterName, **kwargs: self.applyFilter(foo, **kwargs))

    def _add_imageby_lambdas(self):
        # Native CoreImage
        imageByMethods = ['applyTransform', 'clamp']

        # Utility
        imageByMethods += ['transform', 'scale', 'translate', 'rotate']

        def apply(imageByMethod):
            return lambda self, *args: self.applyImageBy(imageByMethod, *args)

        for imageByMethod in imageByMethods:
            setattr(cimg, imageByMethod, apply(imageByMethod))

    @property
    def origin(self):
        return self.extent[0], self.extent[1]

    @property
    def extent(self):
        extent = self._ciimage.extent()
        return extent.origin.x, extent.origin.y, extent.size.width, extent.size.height

    @property
    def size(self):
        return self.extent[2], self.extent[3]

    @property
    def ciimage(self):
        return self._ciimage

    def render(self, dtype=None, alpha=None):
        """
        Render the underlying CIImage to a numpy array.
        """
        if dtype is None:
            dtype = self.dtype

        # Strict dtype checks since not all are supported by CoreImage
        if dtype not in SUPPORTED_DTYPES:
            raise NotImplementedError(
                'Incompatible image type: {}. Must be one of {}.'.format(dtype, SUPPORTED_DTYPES))

        if FLAG_DEBUG:
            print('\nbaking for:', inspect.stack()[1][3])

        if self._dirty:
            if FLAG_DEBUG:
                print('Baking to', dtype)

            # Render
            self._cake = cimg.realize_numpy_from_ciimage(self._ciimage, dtype=dtype)
            self._dirty = False

            # FIXME: render directly to RGBx
            if not alpha:
                self._cake = self._cake[..., :3]

        else:
            if FLAG_DEBUG:
                print('Not dirty, returning existing cake ({})'.format(self._cake is not None))

        return self._cake.copy()

    def save(self, filename):
        """ Save to disk. """
        ext = os.path.splitext(filename)[1].lower()
        supported = ['.jpg', '.jpeg', '.png', '.tiff', '.heif']

        if not any([ext == e for e in supported]):
            warn('{} only supported for now but received {}'.format(supported, ext))

        outURL = NSURL.fileURLWithPath_(filename)
        options = {}

        if ext in ['.jpg', '.jpeg']:
            singletonContext.writeJPEGRepresentationOfImage_toURL_colorSpace_options_error_(self._ciimage,
                                                                                            outURL,
                                                                                            singletonColorspace,
                                                                                            options,
                                                                                            None)

        elif ext in ['.png']:
            singletonContext.writePNGRepresentationOfImage_toURL_format_colorSpace_options_error_(self._ciimage,
                                                                                                  outURL,
                                                                                                  kCIFormatRGBA8,
                                                                                                  singletonColorspace,
                                                                                                  options,
                                                                                                  None)

        elif ext in ['.tiff']:
            singletonContext.writeTIFFRepresentationOfImage_toURL_format_colorSpace_options_error_(self._ciimage,
                                                                                                   outURL,
                                                                                                   kCIFormatRGBA8,
                                                                                                   singletonColorspace,
                                                                                                   options,
                                                                                                   None)

    def average(self):
        return self.areaAverage(extent=self.extent, clamped=False, cropped=False)

    def over(self, image):
        """ Composite on top of another image. """
        assert isinstance(image, cimg)
        return cimg(self._ciimage.imageByCompositingOverImage_(image._ciimage))

    def multiply(self, image):
        src = """
        kernel vec4 _pyci_multiply(__sample a, __sample b) {
            return a*b;
        }
        """
        return self.applyKernel(src, image)

    def crop(self, *args):
        """
        :param args: (x,y,w,h) or (w,h)
        :return: a crop of this image
        """

        # Attempt unpacking from CGRect, NSPoint or tuple
        if len(args) == 1:
            c = args[0]

            if isinstance(c, CGRect):
                args = c.origin.x, c.origin.y, c.size.width, c.size.height

            elif isinstance(c, CGSize):
                args = 0, 0, c.width, c.height

            elif isinstance(c, (tuple, list)):
                args = c

            elif isinstance(c, (float, int)):
                args = c, c

        # Parse crop region
        if len(args) == 2:
            x, y, w, h = 0, 0, args[0], args[1]
        elif len(args) == 4:
            x, y, w, h = args
        else:
            raise RuntimeError('Expecting 2 or 4 tuple or args (origin and size) but received \'{}\''.format(args))

        rect = CGRectMake(x, y, w, h)
        return cimg(self.ciimage.imageByCroppingToRect_(rect))

        return None

    def resize(self, size, preserveAspect=0):
        """
        :param size: scalar or 2-tuple.
        :param preserveAspect: whether to preserve the aspect ratio (scaling down) if `size` is a scalar.
            A value of 0 means no aspect scaling, 1 means fix X axis, 2 fix Y axis.
        :return: a `cimg`
        """

        if not isinstance(size, (tuple, list)):
            sx, sy = size, size

        elif isinstance(size, (tuple, list)) and len(size) == 2:
            sx, sy = [float(s) for s in size]

        else:
            raise RuntimeError('Specify a dimension to use for both axis, or specify them individually')

        sx, sy = float(sx), float(sy)

        if preserveAspect == 1:
            sy = sx * self.size[1] / self.size[0]

        elif preserveAspect == 2:
            sx = sy * self.size[0] / self.size[1]

        return self.scale(sx / self.size[0], sy / self.size[1])

    def applyFilter(self, filterName, clamped=True, cropped=True, **filterInputKeyValues):

        # Clamp image to infinity (repeated boundary conditions)
        inputImage = self.ciimage
        if clamped:
            inputImage = inputImage.imageByClampingToExtent()

        # Massage input keys and values
        filterInputKeyValues = self._validateInputs(filterInputKeyValues)

        # Force set input image with backed CIImage
        filterInputKeyValues['inputImage'] = inputImage

        # Create the filter with specified input parameters
        filter = self._create_filter(filterName, filterInputKeyValues)

        result = filter.outputImage()

        if result is None:
            raise RuntimeError(
                'Internal CoreImage returned nil on outputImage. Make sure all input parameters are set.')

        # Crop image back to input size
        if cropped:
            result = result.imageByCroppingToRect_(self.ciimage.extent())

        return cimg(result)

    def applyKernel(self, kernelSource, *args, **kwargs):

        extent = kwargs['extent'] if 'extent' in kwargs else self.extent
        roi = kwargs['roi'] if 'roi' in kwargs else None
        args = [self._ciimage] + list(args)
        return cimg.fromKernel(kernelSource, extent, roi=roi, args=args)

    def applyImageBy(self, imageByMethod, *imageByArgs):

        type_number = (int, float)  # , long) # python2 only
        # Native CoreImage
        imageBy = {
            'applyTransform': ('imageByApplyingTransform_', [CGAffineTransform]),
            'clamp': ('imageByClampingToExtent', []),
        }

        # Utility
        imageByUtils = {
            'transform': ('imageByApplyingTransform_', [type_number] * 6,
                          lambda a, b, c, d, tx, ty: [CGAffineTransformMake(a, b, c, d, tx, ty)],),
            'scale': ('imageByApplyingTransform_', [type_number] * 2,
                      lambda sx, sy: [CGAffineTransformMakeScale(sx, sy)]),
            'translate': ('imageByApplyingTransform_', [type_number] * 2,
                          lambda tx, ty: [CGAffineTransformMakeTranslation(tx, ty)]),
            'rotate': ('imageByApplyingTransform_', [type_number],
                       lambda angle: [CGAffineTransformMakeRotation(angle)]),
        }
        imageBy.update(imageByUtils)

        # Validate caller
        if imageByMethod not in imageBy:
            raise NotImplementedError('{}'.format(imageByMethod))

        # Validate args
        imageByArgsExp = imageBy[imageByMethod][1]
        if len(imageByArgs) != len(imageByArgsExp):
            raise RuntimeError(
                'Argument count mismatch for {}: expected {} but received {}'.format(imageByMethod, len(imageByArgsExp),
                                                                                     len(imageByArgs)))

        for i, (arg, expected) in enumerate(zip(imageByArgs, imageByArgsExp)):
            if not isinstance(arg, expected):
                raise RuntimeError(
                    'Invalid argument {}: expected type {} but received {}'.format(i, expected, type(arg)))

        # Repackage arguments for utility methods
        if imageByMethod in imageByUtils:
            imageByArgs = imageByUtils[imageByMethod][2](*imageByArgs)

        # Apply
        # FIXME: clamping to extent should be an option

        result = getattr(self._ciimage, imageBy[imageByMethod][0])(*imageByArgs)

        return cimg(result)

    @staticmethod
    def fromFile(filename, useDepth=False, useMatte=False, useDisparity=False,
                 options=None):
        if not os.path.exists(filename) or not os.path.isfile(filename):
            raise IOError('File does not exist: ' + str(filename))

        url = NSURL.fileURLWithPath_(filename)

        options_defaults = {
            'kCIImageApplyOrientationProperty': True,
            'kCIImageCacheHint': False,
            'kCIImageCacheImmediately': False,
            'kCIImageAVDepthData': None,
        }
        options = options if options else options_defaults

        if useDepth:
            options['kCIImageAuxiliaryDepth'] = useDepth
            options['kCIImageApplyOrientationProperty'] = options[
                'kCIImageApplyOrientationProperty'] if 'kCIImageApplyOrientationProperty' in options else True

        elif useDisparity:
            options['kCIImageAuxiliaryDisparity'] = useDisparity
            options['kCIImageApplyOrientationProperty'] = options[
                'kCIImageApplyOrientationProperty'] if 'kCIImageApplyOrientationProperty' in options else True

        elif useMatte:
            options['kCIImageAuxiliaryPortraitEffectsMatte'] = useMatte
            options['kCIImageApplyOrientationProperty'] = options[
                'kCIImageApplyOrientationProperty'] if 'kCIImageApplyOrientationProperty' in options else True

        image = CIImage.imageWithContentsOfURL_options_(url, options)

        if image is None:
            raise RuntimeError(
                'imageWithContentsOfURL return nil for url:\n{}\nand selected options:\n{}\nMake sure image contains depth and/or '
                'matte data when using useDepth=True and useMatte=True'.format(url, options))

        return cimg(image)

    @staticmethod
    def fromCIImage(ciimage, context=None, colorspace=None, dtype=DEFAULT_FORMAT_RENDER):
        ary = cimg.realize_numpy_from_ciimage(ciimage, context=context, colorspace=colorspace, dtype=dtype)
        return cimg(ary, recipe=ciimage)

    @staticmethod
    def fromColor(r, g, b, a=None):
        c = (r, g, b, a) if a else (r, g, b)
        return cimg(CIImage.imageWithColor_(color(c)))

    @staticmethod
    def fromPixel(r, g, b, a=None):
        cimg._kernelCounter += 1

        a = a if a else 1.0

        src = 'kernel vec4 _pyci_frompixel_{}()'.format(cimg._kernelCounter) + '{'
        src += 'return vec4({}, {}, {}, {});'.format(r, g, b, a)
        src += '}'
        return cimg.fromKernel(src, extent=(1, 1))

    @staticmethod
    def fromGenerator(generatorName, crop=None, **filterInputKeyValues):

        # Massage input keys and values
        filterInputKeyValues = cimg._validateInputs(filterInputKeyValues)

        # Create the filter with specified input parameters
        filter = cimg._create_filter(generatorName, filterInputKeyValues)

        result = filter.outputImage()

        if result is None:
            raise RuntimeError('outputImage returned nil')

        result = cimg(result)

        if crop:
            result = result.crop(crop)

        return result

    @staticmethod
    def fromKernel(kernelSource, extent, args=None, roi=None):

        # Prepare kernel inputs
        roi = roi if roi else lambda i, r: r
        args = args if args else []

        # Make sure roi returns a CGRect proper
        roi_rect = lambda index, r: rectify(roi(index, r))

        # Massage extent
        if isinstance(extent, (tuple, list)):
            if len(extent) == 2:
                extent = (0, 0, extent[0], extent[1])
            elif len(extent) == 4:
                pass
            else:
                raise RuntimeError('Expecting 2- or 4-tuple for extent')

        rect = CGRectMake(*extent)

        if FLAG_DEBUG:
            print(kernelSource)

        kernel = CIKernel.kernelWithString_(kernelSource)

        # Parse inputs
        # TODO: add more polymorphism here, e.g., with collections
        inputs = [cimg._validateType(a) for a in args]

        if FLAG_DEBUG:
            for inp in inputs:
                print(type(inp), inp)

        if isinstance(kernel, CIColorKernel):
            if FLAG_DEBUG:
                print('Color kernel')

            result = kernel.applyWithExtent_arguments_(rect, inputs)
        else:
            if FLAG_DEBUG:
                print('Regular kernel')

            result = kernel.applyWithExtent_roiCallback_arguments_(rect, roi_rect, inputs)

        return cimg(result)

    @staticmethod
    def arrayWithCGImage(cg):
        """ Convert to numpy buffer.
        Args:
            cg (CGImageRef): Input CG image.
        Returns:
            CIArray.
        """

        #
        # FIXME: this operation needs to be made as fast as possible
        # FIXME: use renderToBitmap
        # https://developer.apple.com/library/content/qa/qa1509/_index.html

        # Gather CG info
        width = CoreGraphics.CGImageGetWidth(cg)
        height = CoreGraphics.CGImageGetHeight(cg)
        bytesPerRow = CoreGraphics.CGImageGetBytesPerRow(cg)
        bpc = CoreGraphics.CGImageGetBitsPerComponent(cg)
        bpp = CoreGraphics.CGImageGetBitsPerPixel(cg)
        bytesPerPixel = bpp // 8

        # Get actual pixels
        provider = CoreGraphics.CGImageGetDataProvider(cg)
        pixeldata = CoreGraphics.CGDataProviderCopyData(provider)

        # FIXME: this does not seem necessary
        # pixeldata = CFDataGetBytes(pixeldata, (0, CFDataGetLength(pixeldata)), None)

        numComponents = bpp // bpc
        widthExact = bytesPerRow // bytesPerPixel
        # numBytes = len(pixeldata)
        # heightExact = numBytes / (widthExact * bytesPerPixel)

        # FIXME: currently assume either 8 (RGBA8) or 16 (RGBAh) bit per component
        dtype = None
        if bpc == 8:
            dtype = np.uint8

        elif bpc == 16:
            dtype = np.float16

        elif bpc == 32:
            dtype = np.float32

        assert dtype, 'Unsupported bits per component: %d' % bpc

        if FLAG_DEBUG:
            print('Inferred returned data type', dtype)
            print('Bytes per row', bytesPerRow)
            print('Number of bytes = ', len(pixeldata))
            print('Bits per components / pixels', bpc, bpp)
            print('Expected dimensions:  ', width, height)
            print('Calculated dimensions:', widthExact, height)
            print('Number of components:', numComponents)

        ary = np.frombuffer(pixeldata, dtype=dtype)[:height * widthExact * numComponents]
        ary = ary.reshape((height, widthExact, numComponents))

        # Crop any extra padding
        ary = ary[:height, :width, :]

        return ary

    @staticmethod
    def realize_numpy_from_ciimage(ciimage, context=None, colorspace=None, dtype=DEFAULT_FORMAT_RENDER):
        """ Construct a numpy array from a CIImage
        Args:
            context:
            ciimage:
        Returns:
        """
        context = context if context else singletonContext
        colorspace = colorspace if colorspace else context.workingColorSpace()
        format = dtype2formatRGBA[dtype] if dtype in dtype2formatRGBA else None

        if FLAG_DEBUG:
            print('Rendering CIImage to dtype {} ({}) using colorspace {}'.format(dtype, format, colorspace))
            print('Rendering using context', context)

        # FIXME: Temporary workaround since createCGImage_fromRect_format_colorSpace_ does not guarantee the output
        # to be the one specified in its format arguments.
        # ciimage = ciimage.imageByApplyingFilter_('CIPassThroughGeneralFilter')

        ### Method 1: first to CG, then to NumPy
        # Render to numpy
        # cg = context.createCGImage_fromRect_format_colorSpace_(ciimage, ciimage.extent(), format, colorspace)
        # ary = cimg.arrayWithCGImage(cg)

        ### Method 2: (preferred) directly to NumPy
        w, h = int(ciimage.extent().size.width), int(ciimage.extent().size.height)
        buffer_format = np.float32
        outputFormat = kCIFormatRGBAf
        rb = w * 4 * np.dtype(buffer_format).itemsize
        bitmap = NSMutableData.dataWithLength_(h * rb)
        context.render_toBitmap_rowBytes_bounds_format_colorSpace_(ciimage, bitmap, rb, ciimage.extent(),
                                                                   outputFormat, colorspace)
        ary = np.frombuffer(bitmap, dtype=buffer_format)
        ary = ary.reshape((h, w, 4))

        return ary

    @staticmethod
    def create_cgimage_from_numpy(ary):
        # Parse image size, number of components
        h, w = ary.shape[:2]
        numChan = ary.shape[2] if ary.ndim == 3 else 1

        # Extract channels: R, G, B, (A)
        chs = [ary[..., i] for i in range(numChan)]

        # Prepare data info
        bytesPerComponent = ary.dtype.itemsize
        bitsPerComponent = bytesPerComponent * 8
        bytesPerRow = numChan * bytesPerComponent * w
        bitsPerPixel = numChan * bitsPerComponent
        numBytes = h * bytesPerRow

        if FLAG_DEBUG:
            print('bytesPerComponent', bytesPerComponent)
            print('bytesPerRow', bytesPerRow)
            print('bitsPerPixel', bitsPerPixel)
            print('numBytes', numBytes)
            print('numChan', numChan)

        # Interleave channels
        # FIXME: Costly. Is this always necessary?
        if numChan == 3 or numChan == 4:
            interleaved = np.dstack(chs).reshape(h, -1)
            colorspace = singletonColorspace
        else:
            interleaved = np.ascontiguousarray(ary)
            colorspace = CGColorSpaceCreateDeviceGray()

        # Extract buffer and create CGDataProvider
        # buf = np.getbuffer(interleaved)
        # buf = interleaved.tobytes()
        buf = interleaved.data
        # buf = (np.ones(interleaved.size) * 256).astype(interleaved.dtype)

        nsdata = NSData.dataWithBytes_length_(buf, numBytes)
        provider = CGDataProviderCreateWithCFData(nsdata)

        # Specify bitmap flags
        bitmapInfo = kCGBitmapByteOrderDefault

        if numChan == 3:
            bitmapInfo = bitmapInfo  # | kCGImageAlphaNoneSkipLast
        if numChan == 4:
            bitmapInfo = bitmapInfo | 1  # | kCGImageAlphaPremultipliedLast

        # Higher bits per components needs to specify endian explicitly since Quartz defaults to Big endian when not specified
        if ary.dtype.type is np.uint8:
            pass
        elif ary.dtype.type is np.float16:
            bitmapInfo = bitmapInfo | kCGBitmapFloatComponents | kCGBitmapByteOrder16Little
        elif ary.dtype.type is np.float32:
            bitmapInfo = bitmapInfo | kCGBitmapFloatComponents | kCGBitmapByteOrder32Little

        # decode array for the image.
        # If you do not want to allow remapping of the image's color values, pass NULL for the decode array
        decode = None

        # A Boolean value that specifies whether interpolation should occur.
        # The interpolation setting specifies whether Core Graphics should apply a pixel-smoothing algorithm to the image.
        shouldInterpolate = False

        if FLAG_DEBUG:
            print(w, h, bitsPerComponent, bitsPerPixel, bytesPerRow)
            print(colorspace)
            print(bitmapInfo)
            print('provider', provider)
            print('decode', decode)
            print('shouldInterpolate', shouldInterpolate)
            print('renderingIntent', kCGRenderingIntentDefault)

        cg = CGImageCreate(w, h, bitsPerComponent, bitsPerPixel, bytesPerRow, colorspace, bitmapInfo, provider, decode,
                           shouldInterpolate, kCGRenderingIntentDefault)

        assert cg

        return cg

    @staticmethod
    def create_ciimage_from_numpy(ary):
        """
        Returns: The CIImage representation of self.
        """
        if FLAG_DEBUG:
            print('Creating recipe from data', ary.dtype, ary.shape, 'for', inspect.stack()[1][3])

        # CoreImage only supports 32 bit
        if ary.dtype == np.float64:
            ary = ary.astype(np.float32)

        if ary.dtype not in SUPPORTED_DTYPES:
            raise NotImplementedError(
                'Incompatible image type: {}. Must be one of {}.'.format(ary.dtype, SUPPORTED_DTYPES))

        # Parse image size, number of components
        cg = cimg.create_cgimage_from_numpy(ary)
        assert cg, 'CGImageCreate returned nil'

        ci = CIImage.imageWithCGImage_(cg)
        assert ci, 'CIImage.imageWithCGImage returned nil'

        return ci

    @staticmethod
    def _create_filter(filterName, filterInputKeyValues):

        # Support calling via shorthand, e.g., CIAreaAverage -> areaAverage
        if not filterName.startswith('CI'):
            filterName = 'CI' + filterName[0].upper() + filterName[1:]

        # Create an instance of this filter
        filter = CIFilter.filterWithName_(filterName)

        if filter is None:
            # Find closest match and raise
            matches = get_close_matches(filterName, cimg.filters(), n=10, cutoff=0.5)
            matches = ' | '.join(['{}. {}'.format(i + 1, m) for (i, m) in enumerate(matches)])
            msg = 'No filter found by the name: {}. Did you mean:\n\t{}'.format(filterName, matches)
            raise RuntimeError(msg)

        # Parse known input parameters for this filter
        filterAttributes = CIFilter.filterWithName_(filterName).attributes()
        filterInputs = [k for k in filterAttributes if k.startswith('input')]

        # Set inputs
        for k, v in filterInputKeyValues.items():

            # Certain filters require special needs
            if filterName == 'CIQRCodeGenerator' and k == 'inputMessage':
                try:
                    v = NSData.dataWithBytes_length_(bytes(v, 'utf-8'), len(v))
                except TypeError:
                    v = NSData.dataWithBytes_length_(v, len(v))

            try:
                filter.setValue_forKey_(v, k)

            except (AttributeError, KeyError) as e:
                print('{} while setting attribute \'{}\' on \'{}\''.format(e, k, filterName))

                matches = get_close_matches(k, filterInputs, n=5, cutoff=0.7)
                matches = matches if len(matches) > 0 else filterInputs
                matches = ' | '.join(['{}. {}'.format(i + 1, m) for (i, m) in enumerate(matches)])
                print('\tDid you mean: {}'.format(matches))

        return filter

    @staticmethod
    def _validateType(o, identifier=None):

        # Parse collections as CIVector or CIColor
        if isinstance(o, (list, tuple)):
            if identifier and 'color' in identifier.lower():
                return color(o)

            else:
                # Default to vector
                return vector(o)

        elif isinstance(o, cimg):
            # Use backing CIImage
            return o.ciimage

        elif isinstance(o, ndarray):
            if o.size <= 4:
                return cimg._validateType(o.tolist(), identifier=identifier)

            # Reinterpret numpy array as CIImage and use backing image
            return cimg(o).ciimage

        return o

    @staticmethod
    def _validateInputs(filterInputKeyValues):

        # Reformat key names to inputKeyName
        inputKeys = [key for key in filterInputKeyValues]
        for key in inputKeys:
            if not key.startswith('input'):
                keyInput = 'input' + key[0].upper() + key[1:]
                filterInputKeyValues[keyInput] = filterInputKeyValues.pop(key)

        for key in filterInputKeyValues:
            val = filterInputKeyValues[key]
            filterInputKeyValues[key] = cimg._validateType(val, key)

        return filterInputKeyValues


def demo_minimal(filepath):
    """ Minimal example of image filtering using pyci. """

    # Support for most common image file types, including raw.
    img = cimg.fromFile(filepath)
    print(type(img))
    print(img.size)
    print(img.ciimage)

    # List built-in filters
    for i, f in enumerate(cimg.filters()): print('{:3d} {}'.format(i, f))

    # Print more info (including inputs) for a given filter
    print(cimg.inputs['CIGaussianBlur'])

    # Resize the image
    img = img.resize(1024, preserveAspect=1)

    # Rotate the image
    img = img.rotate(30 / 180.0 * np.pi)

    # Apply a filter
    # Note: can use the full filter name "CIGaussianBlur"
    r = 50
    blur = img.gaussianBlur(radius=r)

    # Save to disk
    blur.save(filepath + '.CIGaussianBlur.jpg')

    show([img, blur], title=['input', 'Gaussian blur with radius {}'.format(r)])