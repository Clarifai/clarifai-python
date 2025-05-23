import io
import json
import re
import uuid
from typing import Iterable, List, Tuple, Union, get_args, get_origin

import numpy as np
from clarifai_grpc.grpc.api.resources_pb2 import Audio as AudioProto
from clarifai_grpc.grpc.api.resources_pb2 import Concept as ConceptProto
from clarifai_grpc.grpc.api.resources_pb2 import Frame as FrameProto
from clarifai_grpc.grpc.api.resources_pb2 import Image as ImageProto
from clarifai_grpc.grpc.api.resources_pb2 import Mask as MaskProto
from clarifai_grpc.grpc.api.resources_pb2 import Point as PointProto
from clarifai_grpc.grpc.api.resources_pb2 import Region as RegionProto
from clarifai_grpc.grpc.api.resources_pb2 import Text as TextProto
from clarifai_grpc.grpc.api.resources_pb2 import Video as VideoProto
from PIL import Image as PILImage

__all__ = [
    "Audio",
    "Concept",
    "Frame",
    "Image",
    "Region",
    "Text",
    "Video",
    "MessageData",
    "NamedFieldsMeta",
    "NamedFields",
    "JSON",
]


class MessageData:
    def to_proto(self):
        raise NotImplementedError

    @classmethod
    def from_proto(cls, proto):
        raise NotImplementedError

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        return cls(value)

    def cast(self, python_type):
        if python_type is self.__class__:
            return self
        raise TypeError(f'Incompatible type for {self.__class__.__name__}: {python_type}')


class NamedFieldsMeta(type):
    """Metaclass to create NamedFields subclasses with __annotations__ when fields are specified."""

    def __call__(cls, *args, **kwargs):
        # Check if keyword arguments are types (used in type annotations)
        if kwargs and all(isinstance(v, type) for v in kwargs.values()):
            # Dynamically create a subclass with __annotations__
            name = f"NamedFields({', '.join(f'{k}:{v.__name__}' for k, v in kwargs.items())})"
            return type(name, (cls,), {'__annotations__': kwargs})
        else:
            # Create a normal instance for runtime data
            return super().__call__(*args, **kwargs)


class NamedFields(metaclass=NamedFieldsMeta):
    """A class that can be used to store named fields with values."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{key}={value!r}' for key, value in self.__dict__.items())})"

    def __origin__(self):
        return self

    def __args__(self):
        return list(self.keys())


class JSON:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other

    def __bool__(self):
        return bool(self.value)

    def __repr__(self):
        return f"JSON({self.value})"

    def to_json(self):
        return json.dumps(self.value)

    @classmethod
    def from_json(cls, json_str):
        return cls(json.loads(json_str))

    @classmethod
    def from_value(cls, value):
        return cls(value)

    def cast(self, python_type):
        if not isinstance(self.value, python_type):
            raise TypeError(f'Incompatible type {type(self.value)} for {python_type}')
        return self.value


class Text(MessageData):
    def __init__(self, text: str, url: str = None):
        self.text = text
        self.url = url

    def __eq__(self, other):
        if isinstance(other, Text):
            return self.text == other.text and self.url == other.url
        if isinstance(other, str):
            return self.text == other
        return False

    def __bool__(self):
        return bool(self.text) or bool(self.url)

    def __repr__(self) -> str:
        return f"Text(text={self.text!r}, url={self.url!r})"

    def to_proto(self) -> TextProto:
        return TextProto(raw=self.text or '', url=self.url or '')

    @classmethod
    def from_proto(cls, proto: TextProto) -> "Text":
        return cls(proto.raw, proto.url or None)

    @classmethod
    def from_value(cls, value):
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Text):
            return value
        if isinstance(value, dict):
            return cls(value.get('text'), value.get('url'))
        raise TypeError(f'Incompatible type for Text: {type(value)}')

    def cast(self, python_type):
        if python_type is str:
            return self.text
        if python_type is Text:
            return self
        raise TypeError(f'Incompatible type for Text: {python_type}')


class Concept(MessageData):
    def __init__(self, name: str, value: float = 1.0, id: str = None):
        self.name = name
        self.value = value
        self.id = id or Concept._concept_name_to_id(name)

    @staticmethod
    def _concept_name_to_id(name: str):
        _name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        _name = _name.replace(' ', '-')
        return _name

    def __repr__(self) -> str:
        return f"Concept(id={self.id!r}, name={self.name!r}, value={self.value})"

    def to_proto(self):
        return ConceptProto(id=self.id, name=self.name, value=self.value)

    @classmethod
    def from_proto(cls, proto: ConceptProto) -> "Concept":
        return cls(id=proto.id, name=proto.name, value=proto.value)


class Region(MessageData):
    def __init__(
        self,
        proto_region: RegionProto = None,
        box: List[float] = None,
        concepts: List[Concept] = None,
        mask: Union['Image', PILImage.Image, np.ndarray] = None,
        point: Tuple[float, float, float] = None,
        track_id: str = None,
        text: str = None,
        id: str = None,
    ):
        if proto_region is None:
            self.proto = RegionProto()
            # use setters for init vals
            if box:
                self.box = box
            if concepts:
                self.concepts = concepts
            if mask:
                self.mask = mask
            if point:
                self.point = point
            if track_id:
                self.track_id = track_id
            if text:
                self.text = text
            self.id = id if id is not None else uuid.uuid4().hex
        elif isinstance(proto_region, RegionProto):
            self.proto = proto_region
        else:
            raise TypeError(
                f"Expected type {RegionProto.__name__}, but got type {type(proto_region).__name__}"
            )

    @property
    def id(self):
        return self.proto.id

    @id.setter
    def id(self, value: str):
        self.proto.id = value

    @property
    def text(self):
        return self.proto.data.text.raw

    @text.setter
    def text(self, value: str):
        self.proto.data.text.raw = value

    @property
    def box(self) -> List[float]:
        """[xmin, ymin, xmax, ymax]"""
        bbox = self.proto.region_info.bounding_box
        # x1, y1, x2, y2
        return [bbox.left_col, bbox.top_row, bbox.right_col, bbox.bottom_row]

    @box.setter
    def box(self, value: List[float]):
        if (
            isinstance(value, list)
            and len(value) == 4
            and all(isinstance(val, (int, float)) for val in value)
        ):
            bbox = self.proto.region_info.bounding_box
            bbox.left_col, bbox.top_row, bbox.right_col, bbox.bottom_row = value
        else:
            raise TypeError(f"Expected a list of 4 float values for 'box', but got: {value}")

    @property
    def concepts(self) -> List[Concept]:
        return [Concept.from_proto(proto) for proto in self.proto.data.concepts]

    @concepts.setter
    def concepts(self, value: List[Concept]):
        if isinstance(value, list) and all(isinstance(concept, Concept) for concept in value):
            self.proto.data.concepts.extend([concept.to_proto() for concept in value])
        else:
            raise TypeError(f"Expected a list of 'Concept' for 'concepts', but got: {value}")

    @property
    def mask(self) -> 'Image':
        return Image.from_proto(self.proto.region_info.mask.image)

    @mask.setter
    def mask(self, value: Union['Image', PILImage.Image, np.ndarray]):
        if isinstance(value, PILImage.Image):
            value = Image.from_pil(value)
        elif isinstance(value, np.ndarray):
            value = Image.from_numpy(value)
        elif not isinstance(value, Image):
            raise TypeError(
                f"Expected one of types ['Image', PIL.Image.Image, numpy.ndarray] got '{type(value).__name__}'"
            )
        self.proto.region_info.mask.CopyFrom(MaskProto(image=value.to_proto()))

    @property
    def track_id(self) -> str:
        return self.proto.track_id

    @track_id.setter
    def track_id(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f"Expected str for track_id, got {type(value).__name__}")
        self.proto.track_id = value

    @property
    def point(self) -> Tuple[float, float, float]:
        """[x,y,z]"""
        point = self.proto.region_info.point
        return point.col, point.row, point.z

    @point.setter
    def point(self, value: Tuple[float, float, float]):
        if not isinstance(value, Iterable):
            raise TypeError(f"Expected a tuple of floats, got {type(value).__name__}")
        value = tuple(value)
        if len(value) != 3:
            raise ValueError(f"Expected 3 elements, got {len(value)}")
        if not all(isinstance(v, float) for v in value):
            raise TypeError("All elements must be floats")
        x, y, z = value
        point_proto = PointProto(col=x, row=y, z=z)
        self.proto.region_info.point.CopyFrom(point_proto)

    def __repr__(self) -> str:
        return f"Region(id={self.id},box={self.box or []}, concepts={self.concepts or None}, point={self.point or None}, mask={self.mask or None}, track_id={self.track_id or None})"

    def to_proto(self) -> RegionProto:
        return self.proto

    @classmethod
    def from_proto(cls, proto: RegionProto) -> "Region":
        return cls(proto)


class Image(MessageData):
    def __init__(self, proto_image: ImageProto = None, url: str = None, bytes: bytes = None):
        if proto_image is None:
            proto_image = ImageProto()
        self.proto = proto_image
        # use setters for init vals
        if url:
            self.url = url
        if bytes:
            self.bytes = bytes

    @property
    def url(self) -> str:
        return self.proto.url

    @url.setter
    def url(self, value: str):
        self.proto.url = value

    @property
    def bytes(self) -> bytes:
        return self.proto.base64

    @bytes.setter
    def bytes(self, value: bytes):
        self.proto.base64 = value

    def __repr__(self) -> str:
        attrs = []
        if self.url:
            attrs.append(f"url={self.url!r}")
        if self.bytes:
            attrs.append(f"bytes=<{len(self.bytes)} bytes>")
        return f"Image({', '.join(attrs)})"

    @classmethod
    def from_url(cls, url: str) -> "Image":
        proto_image = ImageProto(url=url)
        return cls(proto_image)

    @classmethod
    def from_pil(cls, pil_image: PILImage.Image, img_format="PNG") -> "Image":
        with io.BytesIO() as output:
            pil_image.save(output, format=img_format)
            image_bytes = output.getvalue()
        proto_image = ImageProto(base64=image_bytes)
        return cls(proto_image)

    @classmethod
    def from_numpy(cls, numpy_image: np.ndarray, img_format="PNG") -> "Image":
        pil_image = PILImage.fromarray(numpy_image)
        with io.BytesIO() as output:
            pil_image.save(output, format=img_format)
            image_bytes = output.getvalue()
        proto_image = ImageProto(base64=image_bytes)
        return cls(proto_image)

    def to_pil(self) -> PILImage.Image:
        if not self.proto.base64:
            raise ValueError("Image has no bytes")
        return PILImage.open(io.BytesIO(self.proto.base64))

    def to_base64_str(self) -> str:
        if not self.proto.base64:
            raise ValueError("Image has no bytes")
        if isinstance(self.proto.base64, str):
            return self.proto.base64
        if isinstance(self.proto.base64, bytes):
            try:
                # trying direct decode (if already a base64 bytes)
                return self.proto.base64.decode('utf-8')
            except UnicodeDecodeError:
                import base64

                return base64.b64encode(self.proto.base64).decode('utf-8')
        else:
            raise TypeError("Expected str or bytes for Image.base64")

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.to_pil())

    def to_proto(self) -> ImageProto:
        return self.proto

    @classmethod
    def from_proto(cls, proto: ImageProto) -> "Image":
        return cls(proto)

    @classmethod
    def from_value(cls, value):
        if isinstance(value, PILImage.Image):
            return cls.from_pil(value)
        if isinstance(value, Image):
            return value
        raise TypeError(f'Incompatible type for Image: {type(value)}')

    def cast(self, python_type):
        if python_type is Image:
            return self
        if python_type is PILImage.Image or python_type is PILImage:
            return self.to_pil()
        if python_type is np.ndarray or get_origin(python_type) is np.ndarray:
            return self.to_numpy()
        raise TypeError(f'Incompatible type for Image: {python_type}')


class Audio(MessageData):
    def __init__(self, proto_audio: AudioProto = None, url: str = None, bytes: bytes = None):
        if proto_audio is None:
            proto_audio = AudioProto()
        self.proto = proto_audio

        if url:
            self.url = url
        if bytes:
            self.bytes = bytes

    @property
    def url(self) -> str:
        return self.proto.url

    @url.setter
    def url(self, value: str):
        self.proto.url = value

    @property
    def bytes(self) -> bytes:
        return self.proto.base64

    @bytes.setter
    def bytes(self, value: bytes):
        self.proto.base64 = value

    @classmethod
    def from_url(cls, url: str) -> "Audio":
        proto_audio = AudioProto(url=url)
        return cls(proto_audio)

    def __repr__(self) -> str:
        attrs = []
        if self.url:
            attrs.append(f"url={self.url!r}")
        if self.bytes:
            attrs.append(f"bytes=<{len(self.bytes)} bytes>")
        return f"Audio({', '.join(attrs)})"

    def to_proto(self) -> AudioProto:
        return self.proto

    def to_base64_str(self) -> str:
        if not self.proto.base64:
            raise ValueError("Audio has no bytes")
        if isinstance(self.proto.base64, str):
            return self.proto.base64
        if isinstance(self.proto.base64, bytes):
            try:
                # trying direct decode (if already a base64 bytes)
                return self.proto.base64.decode('utf-8')
            except UnicodeDecodeError:
                import base64

                return base64.b64encode(self.proto.base64).decode('utf-8')
        else:
            raise TypeError("Expected str or bytes for Audio.base64")

    @classmethod
    def from_proto(cls, proto: AudioProto) -> "Audio":
        return cls(proto)


class Frame(MessageData):
    def __init__(
        self,
        proto_frame: FrameProto = None,
        image: Image = None,
        regions: List[Region] = None,
        time: float = None,
        index: int = None,
    ):
        if proto_frame is None:
            proto_frame = FrameProto()
        self.proto = proto_frame
        # use setters for init vals
        if image:
            self.image = image
        if regions:
            self.regions = regions
        if time:
            self.time = time
        if index:
            self.index = index

    @property
    def time(self) -> float:
        # TODO: time is a uint32, so this will overflow at 49.7 days
        # we should be using double or uint64 in the proto instead
        return self.proto.frame_info.time / 1000.0

    @time.setter
    def time(self, value: float):
        self.proto.frame_info.time = int(value * 1000)

    @property
    def index(self) -> int:
        return self.proto.frame_info.index

    @index.setter
    def index(self, value: int):
        self.proto.frame_info.index = value

    @property
    def image(self) -> Image:
        return Image.from_proto(self.proto.data.image)

    @image.setter
    def image(self, value: Image):
        self.proto.data.image.CopyFrom(value.to_proto())

    @property
    def regions(self) -> List[Region]:
        return [Region(region) for region in self.proto.data.regions]

    @regions.setter
    def regions(self, value: List[Region]):
        self.proto.data.regions.extend([region.proto for region in value])

    def to_proto(self) -> FrameProto:
        return self.proto

    @classmethod
    def from_proto(cls, proto: FrameProto) -> "Frame":
        return cls(proto)


class Video(MessageData):
    def __init__(self, proto_video: VideoProto = None, url: str = None, bytes: bytes = None):
        if proto_video is None:
            proto_video = VideoProto()
        self.proto = proto_video

        if url:
            self.url = url
        if bytes:
            self.bytes = bytes

    @property
    def url(self) -> str:
        return self.proto.url

    @url.setter
    def url(self, value: str):
        self.proto.url = value

    @property
    def bytes(self) -> bytes:
        return self.proto.base64

    @bytes.setter
    def bytes(self, value: bytes):
        self.proto.base64 = value

    @classmethod
    def from_url(cls, url: str) -> "Video":
        proto_video = VideoProto(url=url)
        return cls(proto_video)

    def __repr__(self) -> str:
        attrs = []
        if self.url:
            attrs.append(f"url={self.url!r}")
        if self.bytes:
            attrs.append(f"bytes=<{len(self.bytes)} bytes>")
        return f"Video({', '.join(attrs)})"

    def to_proto(self) -> VideoProto:
        return self.proto

    def to_base64_str(self) -> str:
        if not self.proto.base64:
            raise ValueError("Video has no bytes")
        if isinstance(self.proto.base64, str):
            return self.proto.base64
        if isinstance(self.proto.base64, bytes):
            try:
                # trying direct decode (if already a base64 bytes)
                return self.proto.base64.decode('utf-8')
            except UnicodeDecodeError:
                import base64

                return base64.b64encode(self.proto.base64).decode('utf-8')
        else:
            raise TypeError("Expected str or bytes for Video.base64")

    @classmethod
    def from_proto(cls, proto: VideoProto) -> "Video":
        return cls(proto)


def cast(value, python_type):
    list_type = get_origin(python_type) is list
    if isinstance(value, MessageData):
        return value.cast(python_type)
    if list_type and isinstance(value, np.ndarray):
        return value.tolist()
    if list_type and isinstance(value, list):
        if get_args(python_type):
            inner_type = get_args(python_type)[0]
            return [cast(item, inner_type) for item in value]
        if not isinstance(value, Iterable):
            raise TypeError(f'Expected list, got {type(value)}')
    return value
