import os


class ClarifaiStreamlitCSS(object):
    """ClarifaiStreamlitCSS helps get a consistent style by default for Clarifai provided
    streamlit apps.
    """

    @classmethod
    def insert_default_css(cls, st):
        """Inserts the default style provided in style.css in this folder into the streamlit page

        Example:
          ClarifaiStreamlitCSS.insert_default_css()

        Note:
          This must be placed in both the app.py AND all the pages/*.py files to get the custom styles.
        """
        file_name = os.path.join(os.path.dirname(__file__), "style.css")
        cls.insert_css_file(file_name, st)

    @classmethod
    def insert_css_file(cls, css_file, st):
        """Open the full filename to the css file and insert it's contents the style of the page."""
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    @classmethod
    def buttonlink(cls, st, label, link, target="_parent", style=""):
        """
        This is a streamlit button that will link to another page (or _self if target is _self).
        It is styled to look like the other stButton>button buttons that are created with st.button().

        You must insert_default_css(st) before using on a page.

        Example:
          ClarifaiStreamlitCSS.insert_default_css(st)
          cols = st.columns(4)
          ClarifaiStreamlitCSS.buttonlink(cols[3], "Button", "https://clarifai.com", "_blank")

        Args:
          st: the streamlit package.
          label: the text string to display in the button.
          link: the url to link the button to.
          target: to open the link in same page (_parent) or new tab (_blank).
          style: additional style to apply to the button link.
            Ex: "background-color: rgb(45, 164, 78); color:white;" makes the button green background with white text.
        """
        astyle = ""
        if style:
            astyle = f'style="{style}"'

        st.markdown(
            f'''
    <div class="stButton">
      <a href="{link}" target="{target}" {astyle}>{label}</a>
    </div>
    ''',
            unsafe_allow_html=True,
        )
