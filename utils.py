# Get some utils func from above
def get_image_net_single_image():
    import requests
    import io
    from PIL import Image
    import random

    image_urls = [
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01440764_tench.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01443537_goldfish.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01484850_great_white_shark.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01491361_tiger_shark.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01494475_hammerhead.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01496331_electric_ray.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01498041_stingray.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01514668_cock.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01514859_hen.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01518878_ostrich.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01530575_brambling.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01531178_goldfinch.JPEG",
    ]
    
    image_url = random.choice(image_urls)
    print(f"Downloading image from: {image_url}")

    # Download the image
    try:
        response = requests.get(image_url)
        response.raise_for_status() # Raise an exception for bad status codes
        img_bytes = io.BytesIO(response.content)
        img = Image.open(img_bytes).convert('RGB') # Open image and ensure it's RGB
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except IOError as e:
        print(f"Error opening image: {e}")
        return None

def get_image_net_transform():
    from torchvision import transforms

    # Define transformations for ImageNet
    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the smaller side to 256
        transforms.CenterCrop(224),  # Crop the center 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform
