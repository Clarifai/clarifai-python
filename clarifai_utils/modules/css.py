import os


class ClarifaiStreamlitCSS(object):
  """ ClarifaiStreamlitCSS helps get a consistent style by default for Clarifai provided
  streamlit apps.
  """

  @classmethod
  def insert_default_css(cls, st):
    """ Inserts the default style provided in style.css in this folder into the streamlit page

    Example:
      ClarifaiStreamlitCSS.insert_default_css()

    Note:
      This must be placed in both the app.py AND all the pages/*.py files to get the custom styles.
    """
    file_name = os.path.join(os.path.dirname(__file__), "style.css")
    cls.insert_css_file(file_name, st)

  @classmethod
  def insert_css_file(cls, css_file, st):
    """ Open the full filename to the css file and insert it's contents the style of the page.
    """
    with open(css_file) as f:
      st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
