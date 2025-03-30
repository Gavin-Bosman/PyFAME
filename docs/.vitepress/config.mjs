import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "PyFAME Documentation",
  description: "API documentation for the PyFAME package",
  srcDir: './src',
  outDir: './docsite',
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/reference/index' },
      { text: 'Reference', link: '/reference/reference' },
      { text: 'Changelog', link: '/extras/changelog'}
    ],

    footer: {
      message: 'Released under the <a href="https://github.com/Gavin-Bosman/PyFAME/blob/master/LICENSE.txt">Gnu General Public License Version 3</a>.',
      copyright: 'Copyright @ 2024-present <a href="https://github.com/Gavin-Bosman">Gavin Bosman</a>'
    },

    sidebar: {
      '/reference/': [
        {
          text: 'PyFAME API Reference',
          items: [
            { text: 'Facial Masking', link: '/reference/reference#facial_masking' },
            { text: 'Facial Occlusion', link: '/reference/reference#facial_occlusion' },
            {
              text: 'Facial Color Shifting',
              link: '/reference#color_shifting',
              items: [
                {text: 'face_color_shift()', link: '/reference/reference#face_color_shift'},
                {text: 'face_saturation_shift()', link: '/reference/reference#face_sat_shift'},
                {text: 'face_brightness_shift()', link: '/reference/reference#face_bright_shift'},
  
  
              ]
            }
          ]
        }
      ]
    },
    
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Gavin-Bosman/PyFAME' }
    ]
  }
})
